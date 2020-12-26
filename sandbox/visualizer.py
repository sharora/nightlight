import pyglet
from pyglet import shapes, graphics
from multiprocessing.connection import Listener

#conversion factor
pix2inch = 4
#field with in inches
fieldwidth = 144

#robot size in inches
robotsize = 18

#window, batch, robot definition
sim_window = pyglet.window.Window(pix2inch*fieldwidth, pix2inch*fieldwidth)
batch = graphics.Batch()
robot = shapes.Rectangle(0, 0, robotsize*pix2inch, robotsize*pix2inch, (255,255,255), batch=batch)

#setting center of rotation to be center of the robot
robot._anchor_x = pix2inch*robotsize/2 
robot._anchor_y = pix2inch*robotsize/2
robot.x = fieldwidth/2 * pix2inch
robot.y = fieldwidth/2 * pix2inch
robot.rotation = 45


address = ('localhost', 6000)
listener = Listener(address, authkey=b'Ok Boomer!')
connection = listener.accept()


@sim_window.event
def on_draw():
    sim_window.clear()
    batch.draw()

def update_pos(dt):
    msg = connection.recv()
    robot.x = msg[0]*pix2inch
    robot.y = msg[1]*pix2inch
    robot.rotation = msg[2]


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update_pos, 1/50.0)
    pyglet.app.run()
