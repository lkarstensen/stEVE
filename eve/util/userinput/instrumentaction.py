# pylint: disable=no-member
from typing import Tuple
import numpy as np
import pygame
import serial
import time


class JoyOneDevice:
    def __init__(
        self, action_limits: Tuple[float, float] = (25, 3.14), joystick_id: int = 0
    ) -> None:
        pygame.init()
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(joystick_id)
        self.action_limits = action_limits

    def get_action(self):
        pygame.event.get()
        trans0 = -self.joy.get_axis(1) * self.action_limits[0]
        rot0 = self.joy.get_axis(2) * self.action_limits[1]
        return np.array((trans0, rot0))


class KeyboardOneDevice:
    def __init__(self, actions: Tuple[float, float] = (25, 3.14)) -> None:
        pygame.init()
        self.actions = actions

    def get_action(self):
        trans = 0.0
        rot = 0.0
        pygame.event.get()
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_w] or keys_pressed[pygame.K_UP]:
            trans += self.actions[0]
        if keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]:
            trans -= self.actions[0]
        if keys_pressed[pygame.K_a] or keys_pressed[pygame.K_LEFT]:
            rot += self.actions[1]
        if keys_pressed[pygame.K_d] or keys_pressed[pygame.K_RIGHT]:
            rot -= self.actions[1]
        return np.array((trans, rot))


class KeyboardTwoDevice:
    def __init__(
        self,
        actions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (25, 3.14),
            (25, 3.14),
        ),
    ) -> None:
        pygame.init()
        self.actions = actions

    def get_action(self):
        trans0 = 0.0
        rot0 = 0.0
        trans1 = 0.0
        rot1 = 0.0
        pygame.event.get()
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_UP]:
            trans0 += self.actions[0][0]
        if keys_pressed[pygame.K_DOWN]:
            trans0 -= self.actions[0][0]
        if keys_pressed[pygame.K_LEFT]:
            rot0 += self.actions[0][1]
        if keys_pressed[pygame.K_RIGHT]:
            rot0 -= self.actions[0][1]

        if keys_pressed[pygame.K_z]:
            trans1 += self.actions[1][0]
        if keys_pressed[pygame.K_h]:
            trans1 -= self.actions[1][0]
        if keys_pressed[pygame.K_g]:
            rot1 += self.actions[1][1]
        if keys_pressed[pygame.K_j]:
            rot1 -= self.actions[1][1]
        return np.array(((trans0, rot0), (trans1, rot1)))


class SerialOneDevice:
    
    def __init__(self, com:str, baud:int) -> None:
        pygame.init()
        self.com = com
        self.baud = baud
        self.ser = self.haptic_serial_open()
        self.old_trans = 0
        self.old_rot = 0


    def get_action(self):

        haptic_feedback_set = [ 0 , 0 ]
        trans = 0.0
        rot = 0.0

        haptic_pos , haptic_vel, haptic_force_get = self.haptic_interface( self.ser , haptic_feedback_set )
        pygame.event.get()

        print(haptic_pos)

        trans = haptic_pos[0] - self.old_trans
        rot = haptic_pos[1] - self.old_rot

        self.old_trans = haptic_pos[0]
        self.old_rot = haptic_pos[1]

        return np.array((trans, rot))

    def haptic_serial_open(self):    
        # open serial
        self.ser = serial.Serial('COM13', 115200, timeout=1)
        time.sleep(1)
        return self.ser

    def haptic_interface(self, ser , haptic_feedback_set ):
        haptic_pos = [ 0, 0, 0, 0, 0, 0 ]
        haptic_vel = [ 0, 0, 0, 0, 0, 0 ]
        haptic_force_get = [0, 0]

        # send serial inputs
        # ser.write(bytes("10,10\n", 'utf-8'))
        self.ser.write( bytes( str( haptic_feedback_set[0] ) + ',' + str( haptic_feedback_set[1] ) + '\n' , 'utf-8') )
        time.sleep(0.01)

        isWaiting = ser.inWaiting()
        if ( isWaiting > 0):
            # print( ser.read(ser.inWaiting()) )

            try:
                data_str = ser.read(ser.inWaiting()).decode('ascii')
            except:
                return haptic_pos , haptic_vel , haptic_force_get
            # print(data_str, end='\n')
            time.sleep(0.01)

            haptic_encoder = data_str.split('\r\n')
            he_size = len( haptic_encoder )

            while( he_size > 2 ): # looking for valid data
                he_split = haptic_encoder[he_size-2].split(',') # skip the last entry since it is usually incomplete!
                if len( he_split ) == 14:
                    lin_wire, rot_wire, lin_mcath, rot_mcath, lin_cath, rot_cath, vlin_wire, vrot_wire, vlin_mcath, vrot_mcath, vlin_cath, vrot_cath, lin_pwm, rot_pwm = haptic_encoder[he_size-2].split(',')
                    haptic_pos = [ int(lin_wire), int(rot_wire), int(lin_mcath), int(rot_mcath), int(lin_cath), int(rot_cath) ]
                    haptic_vel = [ int(vlin_wire), int(vrot_wire), int(vlin_mcath), int(vrot_mcath), int(vlin_cath), int(vrot_cath) ]
                    haptic_force_get = [ int(lin_pwm), int(rot_pwm) ]
                    # print( he_split ) # test
                    break
                he_size = he_size - 1

        return haptic_pos , haptic_vel , haptic_force_get
    
class SerialTwoDevice:
    
    def __init__(self, com:str, baud:int) -> None:
        pygame.init()
        self.actions = actions
        self.com = com
        self.baud = baud
        self.ser = self.haptic_serial_open()

        self.old_trans0 = 0
        self.old_rot0 = 0
        self.old_trans1 = 0
        self.old_rot1 = 0



    def get_action(self):

        haptic_feedback_set = [ 0 , 0 ]
        trans0 = 0.0
        rot0 = 0.0
        trans1 = 0.0
        rot1 = 0.0

        haptic_pos , haptic_vel, haptic_force_get = self.haptic_interface( self.ser , haptic_feedback_set )
        pygame.event.get()

        print(haptic_pos)

        trans0 = haptic_pos[0] - self.old_trans0
        rot0 = haptic_pos[1] - self.old_rot0
        trans1 = haptic_pos[2] - self.old_trans1
        rot1 = haptic_pos[3] - self.old_rot1

        self.old_trans0 = haptic_pos[0]
        self.old_rot0 = haptic_pos[1]
        self.old_trans1 = haptic_pos[2]
        self.old_rot1 = haptic_pos[3]

        return np.array((trans0, rot0), (trans1, rot1))

    def haptic_serial_open(self):    
        # open serial
        self.ser = serial.Serial(self.COM, self.baud, timeout=1)
        time.sleep(1)
        return self.ser

    def haptic_interface(self, ser , haptic_feedback_set ):
        haptic_pos = [ 0, 0, 0, 0, 0, 0 ]
        haptic_vel = [ 0, 0, 0, 0, 0, 0 ]
        haptic_force_get = [0, 0]

        # send serial inputs
        # ser.write(bytes("10,10\n", 'utf-8'))
        self.ser.write( bytes( str( haptic_feedback_set[0] ) + ',' + str( haptic_feedback_set[1] ) + '\n' , 'utf-8') )
        time.sleep(0.01)

        isWaiting = ser.inWaiting()
        if ( isWaiting > 0):
            # print( ser.read(ser.inWaiting()) )

            try:
                data_str = ser.read(ser.inWaiting()).decode('ascii')
            except:
                return haptic_pos , haptic_vel , haptic_force_get
            # print(data_str, end='\n')
            time.sleep(0.01)

            haptic_encoder = data_str.split('\r\n')
            he_size = len( haptic_encoder )

            while( he_size > 2 ): # looking for valid data
                he_split = haptic_encoder[he_size-2].split(',') # skip the last entry since it is usually incomplete!
                if len( he_split ) == 14:
                    lin_wire, rot_wire, lin_mcath, rot_mcath, lin_cath, rot_cath, vlin_wire, vrot_wire, vlin_mcath, vrot_mcath, vlin_cath, vrot_cath, lin_pwm, rot_pwm = haptic_encoder[he_size-2].split(',')
                    haptic_pos = [ int(lin_wire), int(rot_wire), int(lin_mcath), int(rot_mcath), int(lin_cath), int(rot_cath) ]
                    haptic_vel = [ int(vlin_wire), int(vrot_wire), int(vlin_mcath), int(vrot_mcath), int(vlin_cath), int(vrot_cath) ]
                    haptic_force_get = [ int(lin_pwm), int(rot_pwm) ]
                    # print( he_split ) # test
                    break
                he_size = he_size - 1

        return haptic_pos , haptic_vel , haptic_force_get