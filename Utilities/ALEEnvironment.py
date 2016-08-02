import sys

''' Set USE_SDL to true to display the screen. ALE must be compilied
   with SDL enabled for this to work. On OSX, pygame init is used to
   proxy-call SDL_main. '''
def initializeSDL(ale):
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

def initializeALEParameters(ale, seed, repeatActionProbability, initSDL = False):
    sys.setrecursionlimit(10000)
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", repeatActionProbability)
    ale.setBool("showinfo", False)

    if initSDL:
        initializeSDL(ale)

