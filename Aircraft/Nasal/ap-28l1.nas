# Disable AP Menu
settimer(func { gui.menuEnable("autopilot", 0) }, 2);

####################
# Power Switch on Remote
####################
var powerOn = func {
	if ( getprop("/an24/AP-28l1/internal/powered") == 1.0 ) {
		setprop("/an24/AP-28l1/internal/power-on-time", getprop("/sim/time/elapsed-sec") + 5 + rand()*10);
	}
	else {
		setprop("/an24/AP-28l1/internal/engaged", 0);
		setprop("/an24/AP-28l1/internal/kv", 0);
	}

}
 setlistener("an24/AP-28l1/internal/powered", powerOn, 0, 0);

####################
# Engage Button on Remote
####################
var engage_ap = func {
	if ( (getprop("an24/AP-28l1/internal/armed") == 1 or getprop("an24/AP-28l1/internal/engaged") == 1) and getprop("an24/AP-28l1/knob_roll-control") == 0.0 ) {
		setprop("an24/AP-28l1/internal/engaged", 1);
		setprop("an24/AP-28l1/internal/kv", 0);
		setprop("an24/AP-28l1/internal/horizon-mode", 0);
		setprop("an24/AP-28l1/internal/target-pitch", getprop("instrumentation/agd-r[1]/indicated-pitch-deg"));
		setprop("an24/AP-28l1/internal/target-bank", 0.0);
		headings_write();
	}
}

####################
# Disengage Button on Yokes
####################
var disengage_ap = func {
		setprop("an24/AP-28l1/internal/engaged", 0);
		setprop("an24/AP-28l1/internal/kv", 0);
		setprop("an24/AP-28l1/internal/horizon-mode", 0);
}

####################
# Horizon Button on Remote
####################
var horizon_button = func {
	if ( getprop("an24/AP-28l1/internal/engaged") == 1 or getprop("an24/AP-28l1/internal/armed") == 1 ) {
		setprop("an24/AP-28l1/internal/engaged", 1);
		setprop("an24/AP-28l1/internal/horizon-mode", 1);
		setprop("an24/AP-28l1/internal/target-pitch", 0.0);
		settimer(engage_kv_mode,10);
	}
}

####################
# Roll Knob on Remote
####################
var roll_knob = func {
	if ( getprop("an24/AP-28l1/sw_roll-mode") != -1 and getprop("an24/AP-28l1/internal/horizon-mode") != 1 ) {
	setprop("an24/AP-28l1/roll-control", getprop("an24/AP-28l1/knob_roll-control") );
	}
}
 setlistener("an24/AP-28l1/internal/horizon-mode", roll_knob, 0, 0);

####################
# KV (Alt Hold) Knob on Remote
####################
var engage_kv_mode = func {
	if ( getprop("an24/AP-28l1/internal/engaged") == 1 ) {
	setprop("/an24/AP-28l1/internal/target-pressure-inhg", getprop("fdm/jsbsim/systems/pitot-static/S1orE1pressure-inhg"));
	setprop("/an24/AP-28l1/internal/kv", 1);
	}
}

# Helper
var headings_write = func {
	setprop("/an24/AP-28l1/internal/target-heading-gpk-52", getprop("an24/GPK-52/indicated-heading-deg"));
	setprop("/an24/AP-28l1/internal/target-heading-gik-1", getprop("an24/GIK-1/indicated-heading"));
	setprop("/an24/AP-28l1/internal/rudder-gik-1-integrator", 0);
}
 setlistener("/an24/AP-28l1/sw_roll-control", headings_write, 0, 0);
 setlistener("/an24/AP-28l1/internal/yaw-mode", headings_write, 0, 0);
