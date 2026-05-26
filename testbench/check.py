import h5py

ruta_dataset_raw = r"C:\Users\santi\OneDrive\Documentos\visualizer\Base_de_datos_GOLD_Ultimate_PCHIP.h5"

print("Abriendo archivo...")
try:
    with h5py.File(ruta_dataset_raw, 'r') as f:
        print("¡Archivo abierto! Esta es su estructura real:")
        # La función visit() escaneará y mostrará absolutamente todas las carpetas y tablas internas
        f.visit(print)
except Exception as e:
    print(f"Error fatal al abrir el archivo: {e}")