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
robot = shapes.Rectangle(-100, -100, robotsize * pix2inch, robotsize * pix2inch, (255, 255, 255), batch=batch)


# debug = []
# debug.append(shapes.Line(42*4,4*114, 4*128, 4*6,1,color=(255,0,0), batch=batch))

obstacles = []
points = []
mapsquares = []
lidarscan = []

#setting center of rotation to be center of the robot
robot._anchor_x = pix2inch*robotsize/2 
robot._anchor_y = pix2inch*robotsize/2


address = ('localhost', 6000)
listener = Listener(address, authkey=b'Ok Boomer!')
connection = listener.accept()


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
                points[i].radius = pix2inch*temp._w*20
            else:
                points.append(shapes.Circle(pix2inch*temp._x[0], pix2inch*temp._x[1],pix2inch*3,color=(255,0,0), batch=batch))
    elif(msg[0] == "map"):
        oc = msg[1]
        length = oc._length
        width = oc._width
        cell2pix = oc._celldim * pix2inch 
        for i in range(length):
            for j in range(width):
                if(oc._oc[length - i - 1][j] == 0):
                    mapsquares.append(shapes.Rectangle(cell2pix*j, cell2pix*i, cell2pix-1, cell2pix - 1, (115,3,252), batch=batch))
    elif(msg[0] == "lidar"):
        ls = msg[1]
        xt = msg[2]
        # length = ls.shape[0]
        # width = ls.shape[1]
        #TODO remove hardcode
        cell2pix = 4*pix2inch
        for i in range(len(ls)):
            for j in range(len(ls[i])):
                xls = ls[i][j][0] + int(xt[0]/4)
                yls = ls[i][j][1] + int(xt[1]/4)
                if(j == len(ls[i]) - 1):
                    lidarscan.append(shapes.Rectangle(cell2pix*xls, cell2pix*yls, cell2pix-1, cell2pix - 1, (255,0,0), batch=batch))
                else:
                    lidarscan.append(shapes.Rectangle(cell2pix*xls, cell2pix*yls, cell2pix-1, cell2pix - 1, (0,255,0), batch=batch))

    else:
        robot.x = msg[0]*pix2inch
        robot.y = msg[1]*pix2inch
        robot.rotation = msg[2]


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update_pos, 1/50.0)
    pyglet.app.run()
