# pylint: disable=no-member
import numpy as np
import pygame

from ...visualisation import SofaPygame

pygame.init()


class VisuManipulator:
    def __init__(self, visu: SofaPygame) -> None:
        self.visu = visu

    def step(self):
        camera_trans = np.array([0.0, 0.0, 0.0])
        pygame.event.get()
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_r]:
            lao_rao = 0
            cra_cau = 0
            if keys_pressed[pygame.K_d]:
                lao_rao += 10
            if keys_pressed[pygame.K_a]:
                lao_rao -= 10
            if keys_pressed[pygame.K_w]:
                cra_cau -= 10
            if keys_pressed[pygame.K_s]:
                cra_cau += 10
            self.visu.rotate(lao_rao, cra_cau)
        else:
            if keys_pressed[pygame.K_w]:
                camera_trans += np.array([0.0, 0.0, 200.0])
            if keys_pressed[pygame.K_s]:
                camera_trans -= np.array([0.0, 0.0, 200.0])
            if keys_pressed[pygame.K_a]:
                camera_trans -= np.array([200.0, 0.0, 0.0])
            if keys_pressed[pygame.K_d]:
                camera_trans = np.array([200.0, 0.0, 0.0])
            self.visu.translate(camera_trans)
        if keys_pressed[pygame.K_e]:
            self.visu.zoom(1000)
        if keys_pressed[pygame.K_q]:
            self.visu.zoom(-1000)
