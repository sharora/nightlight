import pyglet
from pyglet import shapes, graphics
from multiprocessing.connection import Listener

#conversion factor
pix2inch = 4
#field with in inches
fieldwidth = 288
#occupancy grid to inch
oc2inch = 4

#robot size in inches
robotsize = 18

#window, batch, robot definition
sim_window = pyglet.window.Window(pix2inch*fieldwidth, pix2inch*fieldwidth)
batch = graphics.Batch()


#creating occupancy grid
ocgraphic = []
oclen = int(fieldwidth/oc2inch)
ocwidth = int(fieldwidth/oc2inch)
cell2pixel = oc2inch * pix2inch

for i in range(oclen):
    ocgraphic.append([])
    for j in range(ocwidth):
        ocgraphic[i].append(shapes.Rectangle(cell2pixel*j, cell2pixel*i, cell2pixel-1, cell2pixel - 1, (0,0,0), batch=batch))


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
                points[i].radius = max(1,pix2inch*temp._w*20)
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
                    ocgraphic[i][j].color = (115,3,252)
                else:
                    ocgraphic[i][j].color = (0,0,0)
    elif(msg[0] == "lidar"):
        ls = msg[1]
        xt = msg[2]
        for i in range(oclen):
            for j in range(ocwidth):
                if(ocgraphic[i][j].color == [0,255,0]):
                    ocgraphic[i][j].color = (0,0,0)
                elif(ocgraphic[i][j].color == [255,0,0]):
                    ocgraphic[i][j].color = (115,3,252)
        for i in range(len(ls)):
            for j in range(len(ls[i])):
                xls = ls[i][j][0] + int(xt[0]/4)
                yls = ls[i][j][1] + int(xt[1]/4)
                if(j == len(ls[i]) - 1):
                    ocgraphic[yls][xls].color = (255,0,0)
                else:
                    ocgraphic[yls][xls].color = (0,255,0)
    elif(msg[0] == "clear"):
        for i in range(len(ocgraphic)):
            for j in range(len(ocgraphic[0])):
                    ocgraphic[i][j].color = (0,0,0)
    else:
        robot.x = msg[0]*pix2inch
        robot.y = msg[1]*pix2inch
        robot.rotation = msg[2]

if __name__ == '__main__':
    pyglet.clock.schedule_interval(update_pos, 1/50.0)
    pyglet.app.run()
