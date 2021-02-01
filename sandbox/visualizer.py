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
robot = shapes.Rectangle(-100, -100, robotsize*pix2inch, robotsize*pix2inch, (255,255,255), batch=batch)
obstacles = []
points = []

#setting center of rotation to be center of the robot
robot._anchor_x = pix2inch*robotsize/2 
robot._anchor_y = pix2inch*robotsize/2


address = ('localhost', 6000)
listener = Listener(address, authkey=b'Ok Boomer!')
connection = listener.accept()
l = False


@sim_window.event
def on_draw():
    sim_window.clear()
    batch.draw()

def update_pos(dt):
    msg = connection.recv()
    if(msg[0] == "obstacle"):
        for i in range(len(msg[1])):
            temp = msg[1][i]
            obstacles.append(shapes.Circle(pix2inch*temp._x, pix2inch*temp._y, pix2inch*temp._radius,color=(255,0,0), batch=batch))
    elif(msg[0] == "points"):
        for i in range(len(msg[1])):
            temp = msg[1][i]
            if(len(points) == len(msg[1])):
                points[i].x = pix2inch*temp._x[0]
                points[i].y = pix2inch*temp._x[1]
                points[i].radius = pix2inch*temp._w*10
            else:
                points.append(shapes.Circle(pix2inch*temp._x[0], pix2inch*temp._x[1], pix2inch*3,color=(255,0,0), batch=batch))
    else:
        robot.x = msg[0]*pix2inch
        robot.y = msg[1]*pix2inch
        robot.rotation = msg[2]


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update_pos, 1/50.0)
    pyglet.app.run()
