import pyglet
from pyglet import shapes, graphics

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

@sim_window.event
def on_draw():
    sim_window.clear()
    robot.x += 1
    robot.y += 1
    robot.rotation += 1
    batch.draw()

if __name__ == '__main__':
    pyglet.app.run()
