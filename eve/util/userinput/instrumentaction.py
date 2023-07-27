# pylint: disable=no-member
from typing import Tuple
import numpy as np
import pygame


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
        if keys_pressed[pygame.K_UP]:
            trans += self.actions[0]
        if keys_pressed[pygame.K_DOWN]:
            trans -= self.actions[0]
        if keys_pressed[pygame.K_LEFT]:
            rot += self.actions[1]
        if keys_pressed[pygame.K_RIGHT]:
            rot -= self.actions[1]
        return np.array((trans, rot))
