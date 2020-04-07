import matplotlib.pyplot as plt
import numpy as np
import math
import random
from math import cos, sin, pi


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.position = (x, y)

    def change_coord(self, coord, num):

        if coord == "x":
            self.x = num
        elif coord == "y":
            self.y = num
        else:
            print("Error: The change did not take place. you put wrong coord input")

        self.position = (self.x, self.y)

        return

    def rotate(self, center, angle):

        x_rotated = ((self.x - center.x) * cos(angle)) - ((self.y - center.y) * sin(angle))
        y_rotated = ((self.x - center.x) * sin(angle)) + ((self.y - center.y) * cos(angle))

        new_point = Point(x_rotated + center.x, y_rotated + center.y)

        return new_point


def pythagoras(p1, p2):

    x_diff = (p2.x - p1.x)**2
    y_diff = (p2.y - p1.y)**2

    distance = math.sqrt(x_diff + y_diff)

    return distance


# print(center.position)
class RegPolyGen:

    def __init__(self, image_size, radius, number_of_sides):

        self.image_size = image_size
        self.radius = radius
        self.number_of_sides = number_of_sides
        self.center = Point(random.uniform(self.radius, self.image_size - self.radius),
                            random.uniform(self.radius, self.image_size - self.radius))

        self.angle_of_rotation = (360 / number_of_sides)*(pi/180)
        self.list_of_points = []

        self.dict_of_names = {3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon",
                              7: "Heptagon", 8: "Octagon", 9: "Nonagon", 10: "Decagon"}

        if self.number_of_sides <= 10:
            self.name = self.dict_of_names[self.number_of_sides]

    # def rotate(self, point, angle):
    #
    #     x_rotated = ((point.x - self.center.x) * cos(angle)) - ((point.y - self.center.y) * sin(angle))
    #     y_rotated = ((point.x - self.center.x) * sin(angle)) + ((point.y - self.center.y) * cos(angle))
    #
    #     new_point = Point(x_rotated + self.center.x, y_rotated + self.center.y)
    #
    #     return new_point

    def generate_shape(self, rotation_randomization=False):

        first_point = Point(self.center.x, self.center.y + self.radius)
        self.list_of_points.append(first_point)
        next_point = first_point

        for i in range(self.number_of_sides):
            temp_point = self.list_of_points[i].position
            if next_point.rotate(self.center, self.angle_of_rotation).position == temp_point:
                break
            else:
                self.list_of_points.append(next_point.rotate(self.center, self.angle_of_rotation))
                next_point = next_point.rotate(self.center, self.angle_of_rotation)

        if rotation_randomization:
            new_angle = (360)*(pi/180)
            rotation_angle = random.uniform(0, new_angle)
            for z in self.list_of_points:
                new_point = z.rotate(self.center, angle=rotation_angle)
                z.change_coord("x", new_point.x)
                z.change_coord("y", new_point.y)

        return self.list_of_points


def data_set_generator(num_of_shapes, path, image_size=100, rotation_randomiser=False,verbose=1):

    list_of_already_used_numbers = []

    name = ""
    for i in range(num_of_shapes):

        if verbose == 1:
            print("Now creating a new shape...")
        shape = RegPolyGen(image_size, random.uniform(image_size/8, image_size/2), random.randint(3, 10))
        name = shape.name

        if verbose == 1:

            print(f"A new shape has been created! the new shape is\
             a {shape.name } with {shape.number_of_sides} edges and a radius of {shape.radius}")
            print(f"current amount of shapes = {i + 1}")
        list_of_points = shape.generate_shape()

        xs = [k.x for k in list_of_points]
        ys = [k.y for k in list_of_points]

        fig = plt.figure(image_size)
        plt.plot(xs, ys, color="black", linewidth=random.randint(10, 15))

        plt.axis([0, image_size, 0, image_size])
#       plt.show()                                  # Uncomment to see the generated shape

        random_num = random.randint(0, num_of_shapes**2)
        while random_num in list_of_already_used_numbers:
            random_num = random.randint(0, num_of_shapes**2)

        fig.savefig(f"{path}/{name}{random_num}.jpg")

        list_of_already_used_numbers.append(random_num)

        plt.close()
    return


training_path = "D:/PyCharm_Projects/Third_Semester_Programming/dataset3/Training"  # Change this to suit your directory
testing_path = "D:/PyCharm_Projects/Third_Semester_Programming/dataset3/Testing"  # Change this to suit your directory

number_of_samples = 10000

data_set_generator(number_of_samples,training_path)
data_set_generator(number_of_samples,testing_path)


# # ____________________________- ONE SHAPE -___________________________________________
# image_size1 = 100
# shape1 = RegPolyGen(image_size1, 20, 11)
# list_of_points = shape1.generate_shape(rotation_randomization=True)
#
# point_list = []
#
# # for i in list_of_points:
# #     rounded_point = Point(int(round(i.x)), int(round(i.y)))
# #     point_list.append(rounded_point)
#
# xs1 = []
# ys1 = []
#
# for i in list_of_points:
#
#     xs1.append(i.x)
#     ys1.append(i.y)
# ________________________________________________________________________________________________


# matrix = np.full((image_size1, image_size1), 10)
#
# for i in point_list:
#     matrix[i.position] = 0


# for i in range(len(point_list)):
#
#     if (i + 1) == len(point_list):
#         break
#     else:
#         beginning_point = Point(point_list[i].x, point_list[i].y)
#         ending_point = Point(point_list[i+1].x, point_list[i+1].y)
#
#         while beginning_point.position != ending_point.position:
#
#             x_diff = beginning_point.x - ending_point.x
#             y_diff = beginning_point.y - ending_point.y
#
#             difference_diff = abs(x_diff) - abs(y_diff)
#             rate = 1
#
#             if difference_diff > 0 and (y_diff != 0):
#                 rate = abs(int(round(x_diff/y_diff)))
#             elif difference_diff < 0 and (x_diff != 0):
#                 rate = abs(int(round(y_diff/x_diff)))
#
#             if abs(x_diff + y_diff) == 1:
#                 matrix[beginning_point.position] = 0
#                 break
#             elif abs(x_diff + y_diff) == 0:
#                 break
#
#             if x_diff > 0:
#                 if x_diff > y_diff:
#                     beginning_point.change_coord("x", int(beginning_point.x - rate))
#                 else:
#                     beginning_point.change_coord("x", int(beginning_point.x - 1))
#             elif x_diff < 0:
#                 if x_diff > y_diff:
#                     beginning_point.change_coord("x", int(beginning_point.x + rate))
#                 else:
#                     beginning_point.change_coord("x", int(beginning_point.x + 1))
#
#             if y_diff > 0:
#                 if y_diff > x_diff:
#                     beginning_point.change_coord("y", int(beginning_point.y - rate))
#                 else:
#                     beginning_point.change_coord("y", int(beginning_point.y - 1))
#             elif y_diff < 0:
#                 if y_diff > x_diff:
#                     beginning_point.change_coord("y", int(beginning_point.y + 1))
#                 else:
#                     beginning_point.change_coord("y", int(beginning_point.y + rate))
#
#             matrix[beginning_point.position] = 0
#             matrix[(beginning_point.x + 1, beginning_point.y)] = 0
#             matrix[(beginning_point.x, beginning_point.y + 1)] = 0
#             matrix[(beginning_point.x + 1, beginning_point.y + 1)] = 0
#             matrix[(beginning_point.x, beginning_point.y - 1)] = 0
#             matrix[(beginning_point.x - 1, beginning_point.y)] = 0
#             matrix[(beginning_point.x - 1, beginning_point.y - 1)] = 0


# plt.imshow(matrix, cmap="gray", vmin=0, vmax=10)
# plt.plot(xs1, ys1, color="black", linewidth=7)
# plt.axis([0, image_size1, 0, image_size1])
# plt.show()
