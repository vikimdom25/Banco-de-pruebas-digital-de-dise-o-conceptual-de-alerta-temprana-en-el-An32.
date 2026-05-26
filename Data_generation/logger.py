import socket
import csv
import time
import os
import datetime
import xml.etree.ElementTree as ET

# === CONFIGURACIÓN ===
UDP_IP = "127.0.0.1"
UDP_PORT = 5500
PROTOCOLO_XML = "C:\\Users\\santi\\FlightGear\\Downloads\\fgdata_2024_1\\Protocol\\newplayback.xml"

timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
directorio_logs = "C:\\Users\\santi\\OneDrive\\Documentos\\visualizer\\Logs"
os.makedirs(directorio_logs, exist_ok=True)
ARCHIVO_SALIDA = os.path.join(directorio_logs, f"log_datos_{timestamp_str}.csv")

def extraer_encabezados(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        headers = []
        for var in root.findall(".//chunk"):
            name = var.find("name")
            if name is not None and name.text:
                headers.append(name.text)
        return headers
    except Exception as e:
        print(f"[!] Error leyendo el protocolo XML: {e}")
        return []

headers = ["timestamp_ms"] + extraer_encabezados(PROTOCOLO_XML)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(10.0)

print(f"[+] Esperando datos desde FlightGear en {UDP_IP}:{UDP_PORT}")
print(f"[+] Guardando en: {ARCHIVO_SALIDA}")
print(f"[+] Variables detectadas: {headers[1:]}")

start_time = time.time()
paquetes_recibidos = 0

try:
    with open(ARCHIVO_SALIDA, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        while True:
            try:
                data, _ = sock.recvfrom(9000)
                valores = list(map(float, data.decode("utf-8").strip().split(",")))

                elapsed_time_ms = int((time.time() - start_time) * 1000)
                fila = [elapsed_time_ms] + valores
                writer.writerow(fila)
                paquetes_recibidos += 1

            except socket.timeout:
                print("[!] Fin de grabación por inactividad.")
                break

except KeyboardInterrupt:
    print("\n[!] Grabación interrumpida manualmente.")

finally:
    end_time = time.time()
    duracion_total_s = end_time - start_time
    frecuencia_promedio = paquetes_recibidos / duracion_total_s if duracion_total_s > 0 else 0

    print("\n[✔] Fin de la grabación")
    print(f"[•] Paquetes capturados: {paquetes_recibidos}")
    print(f"[•] Duración total: {duracion_total_s:.2f} segundos")
    print(f"[•] Frecuencia de muestreo promedio: {frecuencia_promedio:.2f} Hz")
