
import time
import logging

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie import Crazyflie


def activate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)


def deactivate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 0)


def light_check(scf: SyncCrazyflie):
    activate_led_bit_mask(scf)
    time.sleep(2)
    deactivate_led_bit_mask(scf)
    time.sleep(2)


def arm(scf: SyncCrazyflie):
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)


def take_off(scf: SyncCrazyflie):
    commander = scf.cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3)


def land(scf: SyncCrazyflie):
    commander = scf.cf.high_level_commander

    commander.land(0.0, 2.0)
    time.sleep(2)

    commander.stop()

logging.basicConfig(level=logging.ERROR)
def simple_log(scf, logconf):

    with SyncLogger(scf, logconf) as logger:

        for log_entry in logger:

            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print('[%d][%s]: %s' % (timestamp, logconf_name, data))

            break





uris = {'radio://0/30/2M/E7E7E7E7E7'
    # 'radio://0/0/2M/E7E7E7E7EA',
}
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel



h = 0.0  # remain constant height similar to take off height



if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    with Swarm(uris, factory=CachedCfFactory(rw_cache='./cache')) as swarm:
        print('Connected to  Crazyflies')
        swarm.parallel_safe(light_check)
        print('Light check done')
        # simple_log(swarm, lg_stab)     
        # swarm.reset_estimators()
        print('Estimators have been reset')

        swarm.parallel_safe(arm)
        swarm.parallel_safe(take_off)
        swarm.parallel_safe(run_square_sequence)
        swarm.parallel_safe(land)