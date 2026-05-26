# remote_autostart.nas
# Puente Telnet → Autostart real del avión

var node = props.getNode("/sim/remote/autostart", 1);
node.setBoolValue(0);

setlistener(node, func {
    if (!node.getBoolValue()) return;

    print("[REMOTE] Autostart solicitado");

    if (typeof(autostart) == "hash" and typeof(autostart.startup) == "func") {
        autostart.startup();
        print("[REMOTE] Autostart ejecutado");
    } else {
        print("[REMOTE] Autostart NO disponible aún");
    }

    node.setBoolValue(0);
});
