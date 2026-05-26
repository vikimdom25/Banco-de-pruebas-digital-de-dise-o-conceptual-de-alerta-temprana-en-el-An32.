import sys
import os
import zipfile
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.transforms as mtransforms

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTableView, QSplitter, QHeaderView, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QAbstractTableModel, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont

# ==============================================================================
#  CONFIGURACIÓN (RUTAS)
# ==============================================================================
DEFAULT_ZIP = r"C:\Users\santi\OneDrive\Escritorio\Solid works\log_datos_2026-01-08_17-36-10_procesado.zip"
DEFAULT_META = r"C:\Users\santi\OneDrive\Documentos\visualizer\METADATA_VUELOS_FINAL.csv"
DEFAULT_HDF5 = r"C:\Users\santi\OneDrive\Documentos\visualizer\Base_de_datos_GOLD_Ultimate.h5"

# Umbrales
SAFE_ALTITUDE = 200.0  
IMPACT_ALTITUDE = 50.0 
STRUCTURAL_G = 9.0     

# Umbrales Etiquetado
WARN_G = 1.4
CRIT_ROLL = 80.0
CRIT_DIVE_VZ = -40.0
CRIT_DIVE_GAMMA = -20.0
KEEP_DIVE_VZ = -15.0
KEEP_DIVE_GAMMA = -5.0
RESET_VZ = -5.0
RESET_PITCH = -5.0

# ==============================================================================
#  LÓGICA ML DINÁMICA
# ==============================================================================
def agregar_targets_ml_dinamico(df):
    col_time = next((c for c in df.columns if 'timestamp' in c.lower() or 'time' in c.lower()), None)
    
    if col_time:
        t_raw = df[col_time].values
        t = (t_raw - t_raw[0]) / 1000.0 if np.max(t_raw) > 10000 else t_raw - t_raw[0]
    else:
        t = np.arange(len(df)) * 0.02

    if len(t) > 1:
        dt_real = np.median(np.diff(t))
        if dt_real <= 0 or np.isnan(dt_real): dt_real = 0.02
    else: dt_real = 0.02

    # FUTURE LABEL (5s)
    horizon_frames = int(5.0 / dt_real)
    df['future_label'] = df['label'].shift(-horizon_frames).fillna(df['label'].iloc[-1]).astype(np.int8)
    
    # TIME TO STALL
    stall_mask = (df['label'] == 2).values
    stall_indices = np.where(stall_mask)[0]
    tts = np.full(len(df), 30.0, dtype=np.float32)
    
    if len(stall_indices) > 0:
        next_stall_ptrs = np.searchsorted(stall_indices, np.arange(len(df)))
        valid_mask = next_stall_ptrs < len(stall_indices)
        indices_validos = np.where(valid_mask)[0]
        
        stall_future_idx = stall_indices[next_stall_ptrs[indices_validos]]
        time_stall = t[stall_future_idx]
        time_curr = t[indices_validos]
        tts[indices_validos] = np.clip(time_stall - time_curr, 0.0, 30.0)
        
    df['time_to_stall'] = tts
    return df

# ==============================================================================
#  LÓGICA DE CORTE INTELIGENTE
# ==============================================================================
def detectar_corte_inteligente(df):
    n = len(df)
    if n < 50: return n, "CORTO"
    alt = df['alt'].values; g = df['g_load'].values
    
    if (g > STRUCTURAL_G).any(): return max(0, np.argmax(g > STRUCTURAL_G) - 5), "ESTRUCTURAL"
    
    mask_airborne = alt > SAFE_ALTITUDE
    if not mask_airborne.any(): return n, "NO_DESPEGUE"
    idx_safe = np.argmax(mask_airborne)
    
    mask_crash = alt[idx_safe:] < IMPACT_ALTITUDE
    if mask_crash.any(): return max(0, idx_safe + np.argmax(mask_crash) - 5), "TERRENO"
    
    return n, "RECUPERACION"

def calcular_etiquetas_v7(alpha, g_load, vz, pitch, roll, flaps_val):
    n = len(alpha)
    states = np.zeros(n, dtype=np.int8)
    gamma = pitch - alpha
    stall_limit = (16.0 - (3.0 * flaps_val)) if not np.isscalar(flaps_val) else np.full(n, 16.0 - 3.0*flaps_val)
    event_active = False 
    
    for i in range(n):
        lim = stall_limit[i]
        trig_red = (alpha[i] > lim) or (abs(roll[i]) > CRIT_ROLL)
        trig_dive = (vz[i] < CRIT_DIVE_VZ) or (gamma[i] < CRIT_DIVE_GAMMA)
        
        if trig_red: s, event_active = 2, True
        elif trig_dive: s, event_active = 3, True
        elif event_active:
            if (vz[i] < KEEP_DIVE_VZ) or (gamma[i] < KEEP_DIVE_GAMMA): s = 3
            elif (vz[i] > RESET_VZ) and (pitch[i] > RESET_PITCH) and (not trig_red): s, event_active = 0, False
            else: s = 1 if ((alpha[i] > lim-3) or (g_load[i] > WARN_G)) else 0
        else: s = 1 if ((alpha[i] > lim-3) or (g_load[i] > WARN_G)) else 0
        states[i] = s
    return pd.Series(states).rolling(5, center=True).median().fillna(0).astype(np.int8)

# ==============================================================================
#  WORKER THREAD
# ==============================================================================
class BuilderWorker(QThread):
    progress = pyqtSignal(int); log = pyqtSignal(str); finished = pyqtSignal(str)
    def __init__(self, z, m, h): super().__init__(); self.z, self.m, self.h = z, m, h; self.run_flag = True

    def run(self):
        self.log.emit("⏳ Indexando metadatos...")
        try:
            try: dm = pd.read_csv(self.m, sep=',', decimal='.')
            except: dm = pd.read_csv(self.m, sep=';', decimal=',')
            cn = next((c for c in dm.columns if 'archivo' in c or 'file' in c), None)
            if cn: dm.set_index(dm[cn].astype(str).str.replace('.csv',''), inplace=True)
        except: pass

        with h5py.File(self.h, 'w') as f:
            gd = f.create_group('dynamic_states')
            with zipfile.ZipFile(self.z, 'r') as zf:
                files = sorted([x for x in zf.namelist() if x.endswith('.csv') and '__MACOSX' not in x])
                tot = len(files)
                for i, fn in enumerate(files):
                    if not self.run_flag: break
                    try:
                        with zf.open(fn) as c:
                            df = pd.read_csv(c, sep=';', decimal=',', encoding='utf-8')
                            if df.shape[1]<5: c.seek(0); df = pd.read_csv(c, sep=',', encoding='utf-8')
                    except: continue

                    df_fin = df.copy()
                    cm = {}
                    for c in df.columns:
                        cl = c.lower()
                        if 'alpha' in cl: cm['alpha']=c
                        if 'pitch' in cl: cm['pitch']=c
                        if 'roll' in cl: cm['roll']=c
                        if 'flap' in cl: cm['flaps']=c
                        if 'vertical' in cl or 'vz' in cl: cm['vz']=c
                        if 'alt' in cl: cm['alt']=c
                        if 'nlf' in cl: cm['g']=c
                        elif 'pilot-z' in cl: cm['gb']=c

                    ga = lambda k: df[cm[k]].values if k in cm else np.zeros(len(df))
                    aa, ap, ar, af, av, aalt = ga('alpha'), ga('pitch'), ga('roll'), ga('flaps'), ga('vz'), ga('alt')
                    ag = df[cm['g']].values if 'g' in cm else (-df[cm['gb']].values/32.174 if 'gb' in cm and df[cm['gb']].mean()<-20 else np.ones(len(df)))

                    clean_n = fn.replace('.csv','')
                    mr = dm.loc[clean_n] if clean_n in dm.index else None
                    cut = len(df)
                    if mr is not None and 'duracion_valida' in mr:
                        ct = next((c for c in df.columns if 'time' in c.lower()), None)
                        if ct:
                            tv = df[ct].values; ts = (tv-tv[0])/1000.0 if np.max(tv)>10000 else tv
                            cut = np.sum(ts <= (float(mr['duracion_valida']) + 0.2))
                        else: cut = int(float(mr['duracion_valida'])/0.02)
                    else:
                        dc = pd.DataFrame({'alt':aalt, 'g_load':ag})
                        cut, _ = detectar_corte_inteligente(dc)

                    df_fin = df_fin.iloc[:cut]
                    if len(df_fin)<10: continue

                    aa, ag, av, ap, ar, af = aa[:cut], ag[:cut], av[:cut], ap[:cut], ar[:cut], af[:cut]
                    df_fin['label'] = calcular_etiquetas_v7(aa, ag, av, ap, ar, af)
                    df_fin = agregar_targets_ml_dinamico(df_fin)

                    dset = gd.create_dataset(f"sim_{i:03d}", data=df_fin.select_dtypes(include=[np.number]).values.astype('float32'), compression='gzip')
                    dset.attrs['original_file'] = fn
                    dset.attrs['outcome'] = str(mr['outcome']) if mr is not None else 'UNKNOWN'
                    if mr is not None:
                        for k,v in mr.items(): dset.attrs[f"LHS_{k}"] = str(v)
                    
                    cnames = df_fin.select_dtypes(include=[np.number]).columns.tolist()
                    dset.attrs.create('columns', np.array(cnames, dtype=object), dtype=h5py.special_dtype(vlen=str))
                    self.progress.emit(int((i+1)/tot * 100))
        self.finished.emit("DB Generada")

# ==============================================================================
#  GUI
# ==============================================================================
class PModel(QAbstractTableModel):
    def __init__(self, d): super().__init__(); self.d=d
    def rowCount(self, p=None): return self.d.shape[0]
    def columnCount(self, p=None): return self.d.shape[1]
    def data(self, i, r): return f"{self.d.iloc[i.row(), i.column()]:.3f}" if r==Qt.ItemDataRole.DisplayRole else None
    def headerData(self, c, o, r): return self.d.columns[c] if o==Qt.Orientation.Horizontal and r==Qt.ItemDataRole.DisplayRole else None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("AeroData ML - Ultimate Dashboard v6"); self.resize(1600,1000)
        p=QPalette(); p.setColor(QPalette.ColorRole.Window, QColor(53,53,53)); p.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        p.setColor(QPalette.ColorRole.Base, QColor(25,25,25)); p.setColor(QPalette.ColorRole.AlternateBase, QColor(53,53,53))
        p.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white); p.setColor(QPalette.ColorRole.Button, QColor(53,53,53)); p.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        QApplication.instance().setPalette(p)
        
        self.f=None; self.s=[]; self.idx=0
        w=QWidget(); self.setCentralWidget(w); l=QVBoxLayout(w)
        
        h=QHBoxLayout()
        b1=QPushButton("Construir DB"); b1.clicked.connect(self.run_b); b1.setStyleSheet("background:#8e44ad;color:white;padding:6px")
        b2=QPushButton("Cargar H5"); b2.clicked.connect(self.load_d); b2.setStyleSheet("background:#27ae60;color:white;padding:6px")
        self.li=QLabel("Ready"); self.li.setFont(QFont("Segoe UI",12))
        h.addWidget(b1); h.addWidget(b2); h.addStretch(); h.addWidget(QPushButton("<",clicked=self.prev)); h.addWidget(self.li); h.addWidget(QPushButton(">",clicked=self.next))
        l.addLayout(h)
        
        self.pb=QProgressBar(); self.pb.setVisible(False); l.addWidget(self.pb)
        
        sp=QSplitter(Qt.Orientation.Vertical)
        wg=QWidget(); vl=QVBoxLayout(wg); 
        self.fig, (self.ax1, self.ax2)=plt.subplots(2,1,sharex=True, figsize=(10,8))
        self.fig.patch.set_facecolor('#2c3e50'); self.cv=FigureCanvasQTAgg(self.fig)
        
        self.ax_alt = self.ax1.twinx() 
        self.ax_tts = self.ax2.twinx() 
        
        vl.addWidget(NavigationToolbar2QT(self.cv, wg)); vl.addWidget(self.cv); sp.addWidget(wg)
        
        self.tv=QTableView(); self.tv.setStyleSheet("background:#2c3e50;color:white"); sp.addWidget(self.tv)
        l.addWidget(sp); sp.setSizes([600,300])

    def run_b(self):
        self.wk=BuilderWorker(DEFAULT_ZIP, DEFAULT_META, DEFAULT_HDF5)
        self.wk.progress.connect(self.pb.setValue); self.wk.log.connect(self.li.setText)
        self.wk.finished.connect(lambda m: (self.pb.setVisible(False), QMessageBox.information(self,"Info",m), self.load_h(DEFAULT_HDF5)))
        self.pb.setVisible(True); self.wk.start()

    def load_d(self): 
        f,_=QFileDialog.getOpenFileName(self,"H5","","H5 (*.h5)"); 
        if f: self.load_h(f)
    def load_h(self, p): 
        self.f=h5py.File(p,'r'); self.gd=self.f['dynamic_states']; self.s=sorted(list(self.gd.keys())); self.idx=0; self.upd()

    def upd(self):
        if not self.s: return
        sid=self.s[self.idx]; ds=self.gd[sid]
        cols=[c.decode() if isinstance(c,bytes) else c for c in ds.attrs['columns']]
        df=pd.DataFrame(ds[:], columns=cols)
        
        out=ds.attrs.get('outcome','UNK'); clr="#e74c3c" if "IMPACTO" in str(out) else "#2ecc71"
        self.li.setText(f"{sid} | {out}"); self.li.setStyleSheet(f"color:{clr};font-weight:bold")
        self.tv.setModel(PModel(df))
        
        # Limpieza de Ejes
        self.ax1.clear(); self.ax_alt.clear()
        self.ax2.clear(); self.ax_tts.clear()
        
        # Estilo
        for ax in [self.ax1, self.ax2]: 
            ax.set_facecolor('#34495e'); ax.tick_params(colors='white'); ax.grid(True,alpha=0.1)
        self.ax_alt.tick_params(colors='white'); self.ax_tts.tick_params(colors='cyan')
        
        t = np.arange(len(df))*0.02
        if 'timestamp_ms' in df: t=(df['timestamp_ms']-df['timestamp_ms'].iloc[0])/1000.0
        
        # --- PLOT 1: FÍSICA ---
        ca=next((c for c in df.columns if 'alpha' in c.lower()), None)
        if ca: self.ax1.plot(t, df[ca], color='orange', label='Alpha')
        
        calt=next((c for c in df.columns if 'alt' in c.lower()), None)
        if calt: 
            self.ax_alt.plot(t, df[calt], color='white', alpha=0.3, label='Altitud')
            self.ax_alt.set_ylabel("Altitud", color='white')
        
        # --- PLOT 2: INTELIGENCIA ---
        if 'label' in df:
            l=df['label'].values
            self.ax2.plot(t, l, color='#2ecc71', drawstyle='steps-post', lw=2, label='Clase')
            self.ax2.set_yticks([0,1,2,3]); self.ax2.set_yticklabels(['OK','WARN','STALL','RECUP'])
            
            # --- SEMÁFORO COMPLETO (INCLUYE AMARILLO) ---
            tr=mtransforms.blended_transform_factory(self.ax1.transData, self.ax1.transAxes)
            self.ax1.fill_between(t,0,1,where=(l==1),color='#f1c40f',alpha=0.2,transform=tr) # <-- ¡AQUÍ ESTÁ!
            self.ax1.fill_between(t,0,1,where=(l==2),color='red',alpha=0.3,transform=tr)
            self.ax1.fill_between(t,0,1,where=(l==3),color='cyan',alpha=0.3,transform=tr)

        if 'time_to_stall' in df: 
            self.ax_tts.plot(t, df['time_to_stall'], color='cyan', lw=1.5, ls='--', label='TTS (s)')
            self.ax_tts.set_ylabel("TTS (s)", color='cyan')
            self.ax_tts.set_ylim(0, 35)

        l1, lb1 = self.ax1.get_legend_handles_labels()
        l2, lb2 = self.ax_alt.get_legend_handles_labels()
        self.ax1.legend(l1+l2, lb1+lb2, loc='upper left', facecolor='#2c3e50', labelcolor='white')
        
        self.cv.draw()

    def next(self): self.idx=(self.idx+1)%len(self.s); self.upd()
    def prev(self): self.idx=(self.idx-1)%len(self.s); self.upd()

if __name__=="__main__": app=QApplication(sys.argv); w=MainWindow(); w.show(); sys.exit(app.exec())