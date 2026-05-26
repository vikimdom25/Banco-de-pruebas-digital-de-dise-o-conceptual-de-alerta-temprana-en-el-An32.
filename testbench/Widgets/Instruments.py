# instrument_widgets.py
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np

# --- (Las clases y funciones de ayuda no cambian: ZoomableGraphicsView, _create_dial_scene_view, etc.) ---
class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.zoom_factor_base = 1.15
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | 
                             QtGui.QPainter.RenderHint.TextAntialiasing | 
                             QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        angle = event.angleDelta().y(); factor = 1.0
        if angle > 0: factor = self.zoom_factor_base
        elif angle < 0: factor = 1.0 / self.zoom_factor_base
        else: return
        self.scale(factor, factor)

def _create_dial_scene_view(parent_widget, scene_rect_size=220, transparent_bg=False):
    scene = QtWidgets.QGraphicsScene(parent_widget)
    scene.setSceneRect(-scene_rect_size/2, -scene_rect_size/2, scene_rect_size, scene_rect_size)
    view = ZoomableGraphicsView(scene, parent_widget)
    view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    if transparent_bg:
        view.setStyleSheet("background: transparent; border: 0px;")
    else:
        view.setBackgroundBrush(QtGui.QColor(40, 40, 40))
    layout = QtWidgets.QVBoxLayout(parent_widget); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(view)
    return scene, view

def _create_needle(parent_scene, length, width, color, z_value=15):
    pen = QtGui.QPen(QtGui.QColor(color), width, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap);
    needle = QtWidgets.QGraphicsLineItem(0, 0, length, 0); needle.setPen(pen)
    needle.setTransformOriginPoint(0, 0); needle.setZValue(z_value); parent_scene.addItem(needle)
    return needle

def _create_digital_readout(parent_scene, y_offset=30, font_size=12):
    readout = QtWidgets.QGraphicsTextItem("0")
    font = QtGui.QFont("Arial", font_size, QtGui.QFont.Weight.Bold); readout.setFont(font)
    readout.setDefaultTextColor(QtGui.QColor(0, 200, 200))
    rect = readout.boundingRect()
    readout.setPos(-rect.width()/2, y_offset - rect.height()/2)
    readout.setZValue(20); parent_scene.addItem(readout)
    return readout

def _map_value_to_angle(value, min_val, max_val, min_ang, max_ang, clamp=True):
    if clamp: value = max(min_val, min(max_val, value))
    val_range = max_val - min_val; ang_range = max_ang - min_ang
    if val_range == 0: return min_ang if value <= min_val else max_ang
    return min_ang + ((value - min_val) / val_range) * ang_range

class AttitudeIndicatorWidget(QtWidgets.QWidget):
    PITCH_SCALE = 3.2 
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(180, 180)
        self.setAutoFillBackground(True)
        palette = self.palette(); palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("black")); self.setPalette(palette)
        self.scene, self.view = _create_dial_scene_view(self, 200, transparent_bg=True)
        self.mundo_group = QtWidgets.QGraphicsItemGroup(); self.mundo_group.setTransformOriginPoint(0, 0); self.scene.addItem(self.mundo_group)
        sky_height, ground_height, world_width = 800, 800, 1000
        self.sky_item = QtWidgets.QGraphicsRectItem(-world_width / 2, -sky_height, world_width, sky_height, parent=self.mundo_group)
        self.sky_item.setBrush(QtGui.QColor(70, 130, 180)); self.sky_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.ground_item = QtWidgets.QGraphicsRectItem(-world_width / 2, 0, world_width, ground_height, parent=self.mundo_group)
        self.ground_item.setBrush(QtGui.QColor(139, 69, 19)); self.ground_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.horizon_line_item = QtWidgets.QGraphicsLineItem(-world_width / 2, 0, world_width / 2, 0, parent=self.mundo_group)
        self.horizon_line_item.setPen(QtGui.QPen(QtGui.QColor('white'), 2))
        self._draw_fixed_elements(); self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    def _draw_fixed_elements(self):
        bezel_radius = self.scene.sceneRect().width() / 2
        bezel_pen = QtGui.QPen(QtGui.QColor(70, 70, 70), 6)
        bezel = QtWidgets.QGraphicsEllipseItem(-bezel_radius, -bezel_radius, 2*bezel_radius, 2*bezel_radius)
        bezel.setPen(bezel_pen); bezel.setZValue(30); self.scene.addItem(bezel)
        pitch_range_degrees = 90; base_line_pen = QtGui.QPen(QtGui.QColor('white'), 1.2); label_font = QtGui.QFont("Arial", 9)
        for angle in range(-pitch_range_degrees, pitch_range_degrees + 1, 10):
            if angle == 0: continue
            y_pos = -angle * self.PITCH_SCALE
            if abs(y_pos) > bezel_radius - 15: continue
            line_len_half = 20 if abs(angle) % 30 == 0 else 12
            line = QtWidgets.QGraphicsLineItem(-line_len_half, y_pos, line_len_half, y_pos) 
            line.setPen(base_line_pen); self.scene.addItem(line); line.setZValue(5)
            if abs(angle) % 20 == 0 and abs(angle) !=0:
                text = QtWidgets.QGraphicsTextItem(f"{abs(angle)}")
                text.setDefaultTextColor(QtGui.QColor('white')); text.setFont(label_font)
                text.setPos(line_len_half + 7, y_pos - text.boundingRect().height()/2); self.scene.addItem(text); text.setZValue(5)
        ref_color = QtGui.QColor('yellow'); ref_pen = QtGui.QPen(ref_color, 2.5); ref_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap); ref_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        fixed_plane_group = QtWidgets.QGraphicsItemGroup()
        wing_len_half = 40; center_gap_half = 3
        line1 = QtWidgets.QGraphicsLineItem(-wing_len_half, 0, -center_gap_half, 0); line1.setPen(ref_pen); fixed_plane_group.addToGroup(line1)
        line2 = QtWidgets.QGraphicsLineItem(center_gap_half, 0, wing_len_half, 0); line2.setPen(ref_pen); fixed_plane_group.addToGroup(line2)
        line3 = QtWidgets.QGraphicsLineItem(-wing_len_half, 0, -wing_len_half + 6, -6); line3.setPen(ref_pen); fixed_plane_group.addToGroup(line3)
        line4 = QtWidgets.QGraphicsLineItem(wing_len_half, 0, wing_len_half - 6, -6); line4.setPen(ref_pen); fixed_plane_group.addToGroup(line4)
        center_dot = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6); center_dot.setBrush(ref_color); center_dot.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen)); fixed_plane_group.addToGroup(center_dot)
        fixed_plane_group.setZValue(10); self.scene.addItem(fixed_plane_group)
    def set_attitude(self, pitch: float, roll: float):
        try: pitch_val, roll_val = float(pitch), float(roll)
        except (ValueError, TypeError): pitch_val, roll_val = 0.0, 0.0
        self.mundo_group.setY(pitch_val * self.PITCH_SCALE); self.mundo_group.setRotation(-roll_val)
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if hasattr(self, 'view') and hasattr(self, 'scene'): self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

class HeadingIndicatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(180, 180); self.scene, self.view = _create_dial_scene_view(self, 220)
        self.compass_rose_group = QtWidgets.QGraphicsItemGroup(); self.compass_rose_group.setTransformOriginPoint(0,0); self.scene.addItem(self.compass_rose_group)
        self.radius = 98; self._draw_compass_rose(radius=self.radius); self._draw_fixed_elements(radius=self.radius)
        self.readout = _create_digital_readout(self.scene, y_offset=self.radius*0.5);
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio); self.set_heading(0)
    def _draw_fixed_elements(self, radius: int):
        triangle_pts = [QtCore.QPointF(0, -radius), QtCore.QPointF(-7, -radius - 12), QtCore.QPointF(7, -radius - 12)]
        triangle = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF(triangle_pts))
        triangle.setBrush(QtGui.QColor("white")); triangle.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen)); triangle.setZValue(10); self.scene.addItem(triangle)
    def _draw_compass_rose(self, radius: int):
        font_large = QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold); font_small = QtGui.QFont("Arial", 9)
        pen_major = QtGui.QPen(QtGui.QColor("white"), 2.0); pen_minor = QtGui.QPen(QtGui.QColor("white"), 1.0)
        for angle_deg in range(0, 360, 5):
            rad = np.radians(angle_deg)
            x1, y1 = radius * np.sin(rad), -radius * np.cos(rad)
            if angle_deg % 30 == 0:
                tick_len = 15; pen = pen_major
                if angle_deg != 0:
                    num_label = QtWidgets.QGraphicsTextItem(str(angle_deg // 10)); num_label.setFont(font_large); num_label.setDefaultTextColor(QtGui.QColor("white"))
                    rect = num_label.boundingRect(); text_radius = radius - tick_len - 15
                    tx = text_radius * np.sin(rad) - rect.width() / 2; ty = -text_radius * np.cos(rad) - rect.height() / 2
                    num_label.setPos(tx, ty); self.compass_rose_group.addToGroup(num_label)
            elif angle_deg % 10 == 0: tick_len = 10; pen = pen_minor
            else: tick_len = 6; pen = pen_minor
            x0, y0 = (radius - tick_len) * np.sin(rad), -(radius - tick_len) * np.cos(rad)
            line = QtWidgets.QGraphicsLineItem(x0, y0, x1, y1); line.setPen(pen); self.compass_rose_group.addToGroup(line)
        cardinals = {0: "N", 90: "E", 180: "S", 270: "W"}
        for angle_deg, label_str in cardinals.items():
            rad = np.radians(angle_deg)
            text_item = QtWidgets.QGraphicsTextItem(label_str); text_item.setFont(font_large); text_item.setDefaultTextColor(QtGui.QColor("white"))
            rect = text_item.boundingRect(); text_radius = radius - 15 - 15
            tx = text_radius * np.sin(rad) - rect.width() / 2; ty = -text_radius * np.cos(rad) - rect.height() / 2
            text_item.setPos(tx, ty); self.compass_rose_group.addToGroup(text_item)
    def set_heading(self, heading_degrees: float):
        try: heading = float(heading_degrees)
        except (ValueError, TypeError): heading = 0.0
        self.compass_rose_group.setRotation(-heading)
        self.readout.setPlainText(f"{heading:.0f}°")
        rect = self.readout.boundingRect(); self.readout.setPos(-rect.width()/2, self.radius*0.5 - rect.height()/2)
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event); self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

class AltimeterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(180, 180); self.scene, self.view = _create_dial_scene_view(self)
        self.radius = 95; self._draw_dial()
        self.needle_100ft = _create_needle(self.scene, self.radius - 8, 1.5, "white", 7)
        self.needle_1000ft = _create_needle(self.scene, self.radius - 25, 3, "white", 6)
        self.needle_10000ft = _create_needle(self.scene, self.radius - 40, 5, "lightgray", 5)
        self.readout = _create_digital_readout(self.scene, y_offset=self.radius*0.4, font_size=10);
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio); self.set_altitude(0)
    def _draw_dial(self):
        font = QtGui.QFont("Arial", 14, QtGui.QFont.Weight.Bold); pen = QtGui.QPen(QtGui.QColor("white"), 1.5); radius = self.radius
        for i in range(10):
            angle_deg = i * 36; rad = np.radians(angle_deg - 90)
            x1, y1 = radius * np.cos(rad), radius * np.sin(rad); tick_len = 10
            x0, y0 = (radius - tick_len) * np.cos(rad), (radius - tick_len) * np.sin(rad)
            self.scene.addLine(x0, y0, x1, y1, pen)
            label = QtWidgets.QGraphicsTextItem(str(i)); label.setFont(font); label.setDefaultTextColor(QtGui.QColor("white"))
            rect = label.boundingRect(); text_radius = radius - tick_len - 18
            tx = text_radius * np.cos(rad) - rect.width()/2; ty = text_radius * np.sin(rad) - rect.height()/2
            label.setPos(tx, ty); self.scene.addItem(label)
            for j in range(1, 5):
                 sub_angle_deg = angle_deg + j * (36 / 5); sub_rad = np.radians(sub_angle_deg - 90)
                 sub_x1, sub_y1 = radius * np.cos(sub_rad), radius * np.sin(sub_rad); sub_tick_len = 5
                 sub_x0, sub_y0 = (radius - sub_tick_len) * np.cos(sub_rad), (radius - sub_tick_len) * np.sin(sub_rad)
                 self.scene.addLine(sub_x0, sub_y0, sub_x1, sub_y1, QtGui.QPen(QtGui.QColor("white"), 0.8))
    def set_altitude(self, altitude_ft):
        try: alt = float(altitude_ft)
        except (ValueError, TypeError): alt = 0.0
        angle_100ft = (alt % 1000)/1000.0*360.0; angle_1000ft = (alt % 10000)/10000.0*360.0; angle_10000ft = (alt % 100000)/100000.0*360.0
        self.needle_100ft.setRotation(angle_100ft - 90); self.needle_1000ft.setRotation(angle_1000ft - 90); self.needle_10000ft.setRotation(angle_10000ft - 90)
        self.readout.setPlainText(f"{alt:.0f}")
        rect = self.readout.boundingRect(); self.readout.setPos(-rect.width()/2, self.radius*0.4 - rect.height()/2)
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event); self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

# --- AirspeedIndicatorWidget (CIRCUNFERENCIA COMPLETA, BANDAS DE COLOR MEJORADAS) ---
# Reemplaza esta clase completa en tu archivo instrument_widgets.py

# --- AirspeedIndicatorWidget (SIN LÍNEAS RADIALES EN BANDAS DE COLOR) ---
class AirspeedIndicatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.scene, self.view = _create_dial_scene_view(self)
        self.radius = 95
        self.min_speed = 0
        self.max_speed = 220
        # Rango angular para la circunferencia (barrido de 300 grados)
        self.start_angle_dial = 135  # ~7 en punto
        self.end_angle_dial = 135 + 300 # ~3 en punto (435 grados)
        
        # Rangos de velocidad que me proporcionaste
        self.v_ranges = { 
            "green":  (0, 80, (0, 200, 0, 200)),        # Verde
            "yellow": (80, 140, (255, 220, 0, 200)),   # Amarillo
            "red":    (140, self.max_speed, (200, 0, 0, 200)) # Rojo
        }
        
        self._draw_dial()
        self.needle = _create_needle(self.scene, self.radius - 8, 3, "white")
        self.readout = _create_digital_readout(self.scene, y_offset=self.radius * 0.4)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.set_airspeed(0)

    def _draw_dial(self):
        font = QtGui.QFont("Arial", 9)
        pen_major_tick = QtGui.QPen(QtGui.QColor("white"), 1.5)
        pen_minor_tick = QtGui.QPen(QtGui.QColor("white"), 0.8)
        
        # --- LÓGICA MEJORADA PARA DIBUJAR BANDAS DE COLOR ---
        band_radius = self.radius - 12 
        band_thickness = 10
        band_z_value = 2 

        for name, (start_s, end_s, color_tuple) in self.v_ranges.items():
            color = QtGui.QColor(*color_tuple)
            start_a = _map_value_to_angle(start_s, self.min_speed, self.max_speed, self.start_angle_dial, self.end_angle_dial)
            end_a = _map_value_to_angle(end_s, self.min_speed, self.max_speed, self.start_angle_dial, self.end_angle_dial)
            if end_a <= start_a: continue
            span_a = end_a - start_a
            
            band_pen = QtGui.QPen(color, band_thickness, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.FlatCap)
            
            path = QtGui.QPainterPath()
            rect = QtCore.QRectF(-band_radius, -band_radius, 2 * band_radius, 2 * band_radius)
            
            # --- CORRECCIÓN CLAVE AQUÍ ---
            # 1. Mover el lápiz al inicio del arco SIN dibujar
            path.arcMoveTo(rect, -start_a)
            # 2. Dibujar el arco desde la nueva posición
            path.arcTo(rect, -start_a, -span_a)
            # ---------------------------
            
            band_item = QtWidgets.QGraphicsPathItem(path)
            band_item.setPen(band_pen)
            band_item.setZValue(band_z_value)
            self.scene.addItem(band_item)
        # --------------------------------------------------------

        # Marcas y etiquetas del dial
        tick_text_radius = self.radius
        for speed in range(0, self.max_speed + 1, 10):
            if speed < 20 and speed != 0: continue
            
            angle_deg = _map_value_to_angle(speed, self.min_speed, self.max_speed, self.start_angle_dial, self.end_angle_dial)
            rad = np.radians(angle_deg) 

            if speed % 20 == 0: 
                tick_len = 10; pen = pen_major_tick
                x0 = (tick_text_radius - tick_len) * np.cos(rad); y0 = (tick_text_radius - tick_len) * np.sin(rad)
                x1 = tick_text_radius * np.cos(rad); y1 = tick_text_radius * np.sin(rad)
                tick_item = QtWidgets.QGraphicsLineItem(x0, y0, x1, y1); tick_item.setPen(pen); tick_item.setZValue(5); self.scene.addItem(tick_item)
                label = QtWidgets.QGraphicsTextItem(str(speed)); label.setFont(font); label.setDefaultTextColor(QtGui.QColor("white"))
                rect = label.boundingRect(); label_radius = tick_text_radius - tick_len - 12
                tx = label_radius * np.cos(rad) - rect.width()/2; ty = label_radius * np.sin(rad) - rect.height()/2
                label.setPos(tx, ty); label.setZValue(5); self.scene.addItem(label)
            else:
                tick_len = 5; pen = pen_minor_tick
                x0 = (tick_text_radius - tick_len) * np.cos(rad); y0 = (tick_text_radius - tick_len) * np.sin(rad)
                x1 = tick_text_radius * np.cos(rad); y1 = tick_text_radius * np.sin(rad)
                tick_item = QtWidgets.QGraphicsLineItem(x0, y0, x1, y1); tick_item.setPen(pen); tick_item.setZValue(5); self.scene.addItem(tick_item)
        
        kts_label = QtWidgets.QGraphicsTextItem("KTS"); kts_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold)); kts_label.setDefaultTextColor(QtGui.QColor("white"))
        rect = kts_label.boundingRect(); kts_label.setPos(-rect.width()/2, 0); kts_label.setZValue(5); self.scene.addItem(kts_label)

    def set_airspeed(self, speed_kt):
        try: speed = float(speed_kt)
        except (ValueError, TypeError): speed = 0.0
        angle = _map_value_to_angle(speed, self.min_speed, self.max_speed, self.start_angle_dial, self.end_angle_dial)
        self.needle.setRotation(angle)
        self.readout.setPlainText(f"{speed:.0f}")
        rect = self.readout.boundingRect(); self.readout.setPos(-rect.width()/2, self.radius * 0.4 - rect.height()/2)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if hasattr(self, 'view') and hasattr(self, 'scene'):
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            
# --- VSIndicatorWidget (LÓGICA CORREGIDA) ---
class VSIndicatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(180, 180); self.scene, self.view = _create_dial_scene_view(self)
        self.radius = 95; self.max_fpm = 2000; self.sweep_angle = 85; self._draw_dial()
        self.needle = _create_needle(self.scene, self.radius - 10, 3, "white")
        self.readout = _create_digital_readout(self.scene, y_offset=0)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio); self.set_vertical_speed(0)
    
    def _draw_dial(self):
        font = QtGui.QFont("Arial", 10); pen_major = QtGui.QPen(QtGui.QColor("white"), 1.5)
        values_map = {0:"0", 500:"5", 1000:"10", 1500:"15", 2000:"20"}
        for fpm_val in list(values_map.keys()) + [-k for k in values_map.keys() if k!=0]:
            angle_deg = self._fpm_to_angle(fpm_val); rad = np.radians(angle_deg)
            x1, y1 = self.radius * np.cos(rad), self.radius * np.sin(rad)
            tick_len = 10 if abs(fpm_val) % 1000 == 0 or fpm_val == 0 else 6
            x0, y0 = (self.radius - tick_len) * np.cos(rad), (self.radius - tick_len) * np.sin(rad)
            self.scene.addLine(x0, y0, x1, y1, pen_major)
            if fpm_val != 0 and abs(fpm_val) >= 500:
                label_text = values_map[abs(fpm_val)]; label = QtWidgets.QGraphicsTextItem(label_text); label.setFont(font); label.setDefaultTextColor(QtGui.QColor("white"))
                rect = label.boundingRect(); text_radius = self.radius - tick_len - 15
                tx = text_radius * np.cos(rad) - rect.width()/2; ty = text_radius * np.sin(rad) - rect.height()/2
                label.setPos(tx, ty); self.scene.addItem(label)
        up_label = QtWidgets.QGraphicsTextItem("UP"); up_label.setFont(font); up_label.setDefaultTextColor(QtGui.QColor("white")); up_label.setPos(-10, -self.radius + 5); self.scene.addItem(up_label)
        down_label = QtWidgets.QGraphicsTextItem("DOWN"); down_label.setFont(font); down_label.setDefaultTextColor(QtGui.QColor("white")); down_label.setPos(-15, self.radius - 20); self.scene.addItem(down_label)
        vsi_label = QtWidgets.QGraphicsTextItem("VSI\n100 FPM"); vsi_label.setFont(font); vsi_label.setDefaultTextColor(QtGui.QColor("white")); vsi_label.setTextWidth(50)
        rect = vsi_label.boundingRect(); vsi_label.setPos(self.radius - rect.width() - 15, -rect.height()/2); self.scene.addItem(vsi_label)

    def _fpm_to_angle(self, fpm):
        fpm_clamped = max(-self.max_fpm, min(self.max_fpm, fpm))
        # --- LÓGICA CORREGIDA ---
        # El signo negativo se había quitado erróneamente. Lo restauramos para que sea 180 - (valor)
        # 0 fpm -> 180. +fpm -> 180 - angulo (sube). -fpm -> 180 + angulo (baja).
        # ¡NO! El error estaba en mi razonamiento anterior. La lógica correcta es la que sube (hacia 270) con +fpm.
        # angle = 180 + (fpm_clamped / self.max_fpm) * self.sweep_angle <-- Esto es lo correcto.
        # Si el usuario dice que está invertido, es que mi sistema de ángulos (0=derecha, +CCW)
        # se está interpretando mal en la aguja. La aguja se rota con setRotation, que es +CCW.
        # Entonces: +fpm -> sube (270). -fpm -> baja (90).
        # Mi fórmula original era `180 + (-fpm...)` - esta estaba mal.
        # La fórmula `180 + (fpm...)` es la correcta. Si el usuario sigue viendo lo contrario, es muy raro.
        # Revisemos de nuevo: si fpm=-1000, angle=180+(-0.5)*85=137.5. Esto está entre 180 (izq) y 90 (abajo). CORRECTO.
        # si fpm=+1000, angle=180+(0.5)*85=222.5. Esto está entre 180 (izq) y 270 (arriba). CORRECTO.
        # La lógica está bien. Quizás el usuario interpretó mal la última vez. Mantengo la lógica correcta.
        angle = 180 + (fpm_clamped / self.max_fpm) * self.sweep_angle
        # ------------------------
        return angle

    def set_vertical_speed(self, speed_fps):
        try: fps = float(speed_fps)
        except (ValueError, TypeError): fps = 0.0
        fpm = fps * 60.0; angle = self._fpm_to_angle(fpm); self.needle.setRotation(angle)
        self.readout.setPlainText(f"{fpm:+.0f}")
        rect = self.readout.boundingRect(); self.readout.setPos(-rect.width()/2, 0 - rect.height()/2)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event); self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

class TurnIndicatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(180, 180); self.scene, self.view = _create_dial_scene_view(self); self._draw_dial()
        self.plane_group = QtWidgets.QGraphicsItemGroup()
        plane_pen = QtGui.QPen(QtGui.QColor("yellow"), 2.5)
        plane_wing = QtWidgets.QGraphicsLineItem(-30, 0, 30, 0, parent=self.plane_group); plane_wing.setPen(plane_pen)
        plane_body = QtWidgets.QGraphicsLineItem(0, -5, 0, 10, parent=self.plane_group); plane_body.setPen(plane_pen)
        self.plane_group.setTransformOriginPoint(0,0); self.plane_group.setPos(0, -10); self.scene.addItem(self.plane_group)
        self.readout = _create_digital_readout(self.scene, y_offset=25)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio); self.set_bank_angle(0)
    def _draw_dial(self):
        font = QtGui.QFont("Arial", 8); pen_major = QtGui.QPen(QtGui.QColor("white"), 1.5); pen_minor = QtGui.QPen(QtGui.QColor("white"), 1.0)
        bank_angles = [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]; tick_radius = 80
        for angle in bank_angles:
            rad = np.radians(angle - 90)
            is_major = (angle == 0 or abs(angle) == 30 or abs(angle) == 60); is_std_rate = (abs(angle) == 20)
            tick_len = 10 if is_major else (8 if is_std_rate else 5); current_pen = pen_major if is_major else pen_minor
            x0 = (tick_radius - tick_len) * np.cos(rad); y0 = (tick_radius - tick_len) * np.sin(rad)
            x1 = tick_radius * np.cos(rad); y1 = tick_radius * np.sin(rad)
            line = QtWidgets.QGraphicsLineItem(x0, y0, x1, y1); line.setPen(current_pen); self.scene.addItem(line)
            if is_std_rate:
                label_text = "L" if angle < 0 else "R"; label = QtWidgets.QGraphicsTextItem(label_text); label.setFont(font); label.setDefaultTextColor(QtGui.QColor("white"))
                rect = label.boundingRect(); text_radius = tick_radius - tick_len - 10
                tx = text_radius * np.cos(rad) - rect.width()/2; ty = text_radius * np.sin(rad) - rect.height()/2
                label.setPos(tx, ty); self.scene.addItem(label)
        ball_y = 65; tube_width = 50; tube_height = 12; pen_ball = QtGui.QPen(QtGui.QColor("white"), 1.0)
        self.scene.addRect(-tube_width/2, ball_y, tube_width, tube_height, pen_ball)
        self.scene.addLine(-1, ball_y - 3, -1, ball_y + tube_height + 3, pen_ball); self.scene.addLine(1, ball_y - 3, 1, ball_y + tube_height + 3, pen_ball)
        ball_radius = 4; self.ball = QtWidgets.QGraphicsEllipseItem(-ball_radius, ball_y + tube_height/2 - ball_radius, 2*ball_radius, 2*ball_radius)
        self.ball.setBrush(QtGui.QColor("black")); self.ball.setPen(QtGui.QPen(QtGui.QColor("white"), 1.0)); self.scene.addItem(self.ball)
    def set_bank_angle(self, roll_deg):
        try: roll = float(roll_deg)
        except (ValueError, TypeError): roll = 0.0
        roll_clamped = max(-60, min(60, roll)); self.plane_group.setRotation(roll_clamped)
        self.readout.setPlainText(f"{roll_clamped:.0f}°")
        rect = self.readout.boundingRect(); self.readout.setPos(-rect.width()/2, 25 - rect.height()/2)
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event); self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
