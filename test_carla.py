import sys
import glob
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
c = carla.VehicleControl()
print("Gear:", c.gear)
print("Hand Brake:", c.hand_brake)
print("Manual Gear Shift:", c.manual_gear_shift)
