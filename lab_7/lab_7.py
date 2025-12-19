from os import system
import cv2
import numpy as np
import pdb
import math
import time
from copy import copy, deepcopy
import itertools
import sys
import profiling  # Импортируем модуль профилирования


class Point:
    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def __repr__(self):
        return f"[x : {self.x}, y : {self.y}]"

    def __str__(self):
        return f"x : {self.x}, y : {self.y}"

    def convert2Int(self):
        self.x = int(self.x)
        self.y = int(self.y)


class Circle:
    def __init__(self, point, radius, name):
        self.name = str(name)
        self.point = point
        self.radius = int(radius)  # it must be int for the circle

    def __repr__(self):
        return f"nameCircle : {self.name}, circleCenter : {self.point}, radius : {self.radius}"

    def __str__(self):
        return f"nameCircle : {self.name}, circleCenter : {self.point}, radius : {self.radius}"


class Event:
    def __init__(self, circle, pointEvent, typeEvent):
        self.name = circle.name + typeEvent
        self.circle = circle
        self.pointEvent = pointEvent
        self.typeEvent = typeEvent
        self.point = None

    def __repr__(self):
        return f"nameEvent : {self.name}, circle : {self.circle}, pointEvent : {self.pointEvent}, typeEvent : {self.typeEvent}, point : {self.point}"

    def __str__(self):
        return f"nameEvent : {self.name}, circle : {self.circle}, pointEvent : {self.pointEvent}, typeEvent : {self.typeEvent}, point : {self.point}"


def parseInput(file1):
    Lines = file1.readlines()

    data = Lines[0].split()
    x = Point(data[0], data[1])
    data = Lines[1].split()
    y = Point(data[0], data[1])

    count = 0
    circles = []
    for line in Lines[2:]:
        data = line.split()
        circles.append(Circle(Point(data[0], data[1]), data[2], count))
        print("Point {}: {}".format(count, line.strip()))
        count += 1

    y.convert2Int()
    x.convert2Int()
    return y, x, circles


def draw(x, y, circles, scaleFactor=50):
    background = np.zeros((x.y * scaleFactor, y.y * scaleFactor, 3), dtype=np.uint8)
    background[:] = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2

    # fontScale
    fontScale = 1

    # Draw rectangle
    if x.x != 0 or y.x != 0:
        print(f"x.x: {x.x}")
        print(f"x.y: {x.y}")
        print(f"y.x: {y.x}")
        print(f"y.y: {y.y}")
        # top-left, bottom-right
        posForRect = int(y.x * scaleFactor), int(0 * scaleFactor), int(y.y * scaleFactor), int(
            (x.y - x.x) * scaleFactor)
        cv2.rectangle(background, posForRect, (255, 255, 0), 5)

    # Draw a circle with blue line borders of thickness of 2 px
    for circle in circles:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        position = (int(circle.point.x * scaleFactor), int((x.y - circle.point.y) * scaleFactor))
        cv2.circle(background, position, circle.radius * scaleFactor, color, thickness)
        cv2.circle(background, position, 1, color, 5)
        cv2.putText(background, circle.name, (position[0] + 10, position[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    color, 4, cv2.LINE_AA)

    cv2.imshow("background", background)
    cv2.waitKey(0)

    for i in range(len(circles)):
        for j in range(len(circles)):
            intersectionPoints = computeIntersection(circles[i], circles[j], scaleFactor)

            if intersectionPoints is not None:
                for p in intersectionPoints:
                    background = cv2.circle(background, (int(p.x), int((x.y * scaleFactor - p.y))), 1, (0, 0, 255), 10)

    cv2.imshow("background", background)
    cv2.waitKey(0)


def screenShotPlaneSweep(x, y, circles, event, intersection=None, animation=True, scaleFactor=50,
                         intersectionPointsList=None):
    if animation:

        if intersectionPointsList is not None:
            intersectionPointsList = list(itertools.chain.from_iterable(intersectionPointsList))
            if len(intersectionPointsList) > 0:

                background = np.zeros((x.y * scaleFactor, y.y * scaleFactor, 3), dtype=np.uint8)

                # Line thickness of 2 px
                thickness = 2

                # fontScale
                fontScale = 1

                # Draw rectangle
                if x.x != 0 or y.x != 0:
                    # top-left, bottom-right
                    posForRect = int(y.x * scaleFactor), int(0 * scaleFactor), int(y.y * scaleFactor), int(
                        (x.y - x.x) * scaleFactor)
                    cv2.rectangle(background, posForRect, (255, 255, 0), 5)

                # Draw a circle with blue line borders of thickness of 2 px
                for circle in circles:
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    position = (int(circle.point.x * scaleFactor), int((x.y - circle.point.y) * scaleFactor))
                    cv2.circle(background, position, circle.radius * scaleFactor, color, thickness)
                    cv2.circle(background, position, 1, color, 5)
                    cv2.putText(background, circle.name, (position[0] + 10, position[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, color, 4, cv2.LINE_AA)

                for p in intersectionPointsList:
                    background = cv2.circle(background, (int(p.x * scaleFactor), int((x.y - p.y) * scaleFactor)), 1,
                                            (0, 0, 255), 10)

                print(intersectionPointsList)
                cv2.imshow("background", background)
                cv2.waitKey(0)
                return

        background = np.zeros((x.y * scaleFactor, y.y * scaleFactor, 3), dtype=np.uint8)

        # Line thickness of 2 px
        thickness = 2

        # fontScale
        fontScale = 1

        # Draw rectangle
        if x.x != 0 or y.x != 0:
            # top-left, bottom-right
            posForRect = int(y.x * scaleFactor), int(0 * scaleFactor), int(y.y * scaleFactor), int(
                (x.y - x.x) * scaleFactor)
            cv2.rectangle(background, posForRect, (255, 255, 0), 5)

        # Draw a circle with blue line borders of thickness of 2 px
        for circle in circles:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            position = (int(circle.point.x * scaleFactor), int((x.y - circle.point.y) * scaleFactor))
            cv2.circle(background, position, circle.radius * scaleFactor, color, thickness)
            cv2.circle(background, position, 1, color, 5)
            cv2.putText(background, circle.name, (position[0] + 10, position[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        color, 4, cv2.LINE_AA)

        cv2.line(background, (int(event.pointEvent * scaleFactor), 0),
                 (int(event.pointEvent * scaleFactor), int(x.y * scaleFactor)), (0, 255, 0), thickness=2)
        if int(event.pointEvent * scaleFactor) <= int(x.y * scaleFactor / 2):
            pos = (int(event.pointEvent * scaleFactor + 20), int((x.y * scaleFactor) / 2))
        else:
            pos = (int(event.pointEvent * scaleFactor - 130), int((x.y * scaleFactor) / 2))
        cv2.putText(background, event.typeEvent, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 4,
                    cv2.LINE_AA)
        cv2.imshow("background", background)
        cv2.waitKey(0)

        if intersection is not None:
            for spa in intersection:
                for p in intersection[spa]["points"]:
                    background = cv2.circle(background, (int(p.x), int(x.y * scaleFactor - p.y)), 1, (0, 0, 255), 10)

            cv2.imshow("background", background)
            cv2.waitKey(0)


def computeIntersection(circle_a, circle_b, scaleFactor):
    x0, y0, r0 = circle_a.point.x * scaleFactor, circle_a.point.y * scaleFactor, circle_a.radius * scaleFactor
    x1, y1, r1 = circle_b.point.x * scaleFactor, circle_b.point.y * scaleFactor, circle_b.radius * scaleFactor

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return [Point(x3, y3), Point(x4, y4)]


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def findIntersection(sweepline, index, circles, scaleFactor, sweepElem=None):
    intersectionPointsUp = []
    intersectionPointsBelow = []

    if len(sweepline) > 1:
        if sweepElem is None:
            up = sweepline[index]["up"]
            below = sweepline[index]["below"]

            if up is not None:
                intersectionPointsUp = computeIntersection(circles[index], circles[up], scaleFactor)
                if intersectionPointsUp is None:
                    intersectionPointsUp = []

            if below is not None:
                intersectionPointsBelow = computeIntersection(circles[index], circles[below], scaleFactor)
                if intersectionPointsBelow is None:
                    intersectionPointsBelow = []

        else:
            intersectionPointsUp = computeIntersection(circles[index], circles[sweepElem], scaleFactor)
            if intersectionPointsUp is None:
                intersectionPointsUp = []

    return [intersectionPointsUp, intersectionPointsBelow]


def deleteElementFromSweepLine(sweepline, index):
    up = sweepline[index]["up"]
    below = sweepline[index]["below"]

    if up is not None:
        if below is not None:
            sweepline[up]["below"] = below
        else:
            sweepline[up]["below"] = None

    if below is not None:
        if up is not None:
            sweepline[below]["up"] = up
        else:
            sweepline[below]["up"] = None

    del sweepline[index]

    global rootIndex
    if index == rootIndex:
        if len(sweepline) > 0:
            rootIndex = next(iter(sweepline))
        else:
            rootIndex = 0


rootIndex = 0


def updateSweepLine(sweepline, currentEvent, circles):
    if len(sweepline) == 0:
        global rootIndex
        rootIndex = int(currentEvent["obj"].circle.name)
        sweepline[int(currentEvent["obj"].circle.name)] = {
            "up": None,
            "me": rootIndex,
            "below": None,
        }
    else:

        currentCircle = sweepline[rootIndex]
        eventCircleY = currentEvent["obj"].circle.point.y

        while True:

            # If the sweep line has only 1 element
            if currentCircle["up"] is None and currentCircle["below"] is None:
                swapCircleMe = circles[currentCircle["me"]].point.y

                if eventCircleY > swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": None,
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                else:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["me"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": None,
                    }

                    # update the current sweep circle that is below
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)

                break  # we exit immediately

            # If the currentCircle of the sweep line has a None below circle
            elif currentCircle["up"] is None:
                swapCircleMe = circles[currentCircle["me"]].point.y
                swapCircleYBelow = circles[currentCircle["below"]].point.y

                if swapCircleYBelow <= eventCircleY <= swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["me"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["below"],
                    }

                    tmp = sweepline[currentCircle["me"]]["below"]
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["up"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY > swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": None,
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["me"],
                    }

                    # update the current sweep circle that is below the new circle
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                    break

                else:
                    # we look at below
                    currentCircle = sweepline[currentCircle["below"]]

            # If the currentCircle of the sweep line has a None up circle
            elif currentCircle["below"] is None:
                swapCircleYUp = circles[currentCircle["up"]].point.y
                swapCircleMe = circles[currentCircle["me"]].point.y

                if swapCircleMe <= eventCircleY <= swapCircleYUp:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["up"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    tmp = sweepline[currentCircle["me"]]["up"]
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["below"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY < swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["me"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": None,
                    }

                    # update the current sweep circle that is up
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    break

                else:
                    # we look at up
                    currentCircle = sweepline[currentCircle["up"]]

            else:
                swapCircleYUp = circles[currentCircle["up"]].point.y
                swapCircleMe = circles[currentCircle["me"]].point.y
                swapCircleYBelow = circles[currentCircle["below"]].point.y

                if swapCircleMe <= eventCircleY <= swapCircleYUp:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["up"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["me"],
                    }

                    # update the current sweep circle that is below
                    tmp = sweepline[currentCircle["me"]]["up"]
                    sweepline[currentCircle["me"]]["up"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["below"] = int(currentEvent["obj"].circle.name)
                    break

                elif swapCircleYBelow <= eventCircleY <= swapCircleMe:
                    # create the new circle
                    sweepline[int(currentEvent["obj"].circle.name)] = {
                        "up": currentCircle["me"],
                        "me": int(currentEvent["obj"].circle.name),
                        "below": currentCircle["below"],
                    }

                    tmp = sweepline[currentCircle["me"]]["below"]
                    sweepline[currentCircle["me"]]["below"] = int(currentEvent["obj"].circle.name)
                    sweepline[tmp]["up"] = int(currentEvent["obj"].circle.name)
                    break

                elif eventCircleY > swapCircleYUp:
                    currentCircle = sweepline[currentCircle["up"]]

                elif eventCircleY < swapCircleYBelow:
                    currentCircle = sweepline[currentCircle["below"]]


def planeSweepPacking(x, y, circles, animation, scaleFactor):
    events = []
    for circle in circles:
        events.append(Event(circle, circle.point.x - circle.radius, "LEFT"))
        events.append(Event(circle, circle.point.x + circle.radius, "RIGHT"))

    print("\nNot ordered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")

    # A way that can be fastest, especially if your list has a lot
    # of records, is to use operator.attrgetter("count")
    try:
        import operator
    except ImportError:
        keyfun = lambda x: x.count  # use a lambda if no operator module
    else:
        keyfun = operator.attrgetter("pointEvent")  # use operator since it's faster than lambda

    events.sort(key=keyfun, reverse=False)  # sort in-place

    print("\nOrdered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")
    print("\n")

    '''
    # example of eventsQueue
    eventsQueue = {
        "event1" : {
            "obj" : event1,
            "before" : "event0",
            "after" : "event2",
        },

        "event2" :{
            "obj" : event2,
            "before" : "event1",
            "after" : "event3",
        }
    }

    # example of sweepline
    sweepline = {
        0 : {               # name of this circle
            "up" : None,    # name of upper circle
            "me" : 0,       # name of this circle
            "below" : 2,    # name of lower circle
        },

        2 :{
            "up" : 0,
            "me" : 2,
            "below" : 1,
        }

        1 :{
            "up" : 2,
            "me" : 1,
            "below" : None,
        }
    }
    '''

    sweepline = {}

    # We set the priority through a dictionary of events
    eventsQueue = {}
    eventsQueue[events[0].name] = {"obj": events[0], "before": None, "after": events[1].name}

    for i in range(1, len(events[1::])):
        eventsQueue[events[i].name] = {"obj": events[i], "before": events[i - 1].name, "after": events[i + 1].name}

    eventsQueue[events[-1].name] = {"obj": events[-1], "before": events[-2].name, "after": None}

    # Swap line among the events
    count = 0
    currentEvent = eventsQueue[events[0].name]
    while True:
        print(f"Iteration: {count}")
        if animation:
            print(f'Name current obj: {currentEvent["obj"].name}')
            print(eventsQueue)
            pretty(eventsQueue)  # pretty view of data
            print(f"Length eventsQueue: {len(eventsQueue)} \n")

        # LEFT EVENT
        if currentEvent["obj"].typeEvent == "LEFT":

            if int(currentEvent["obj"].circle.name) not in sweepline:
                updateSweepLine(sweepline, currentEvent, circles)

        # LEFT RECTANGLE EVENT
        if currentEvent["obj"].typeEvent == "LEFTRECTANGLE":

            if int(currentEvent["obj"].circle.name) not in sweepline:
                updateSweepLine(sweepline, currentEvent, circles)

        print(f'Type of the event: {currentEvent["obj"].typeEvent}')
        print(f'Check intersection up and below the circle {currentEvent["obj"].circle.name}')
        flagInter = False
        intersectionPointsList = findIntersection(sweepline, int(currentEvent["obj"].circle.name), circles,
                                                  scaleFactor=1)
        if (len(intersectionPointsList[0]) != 0) or (len(intersectionPointsList[1]) != 0):
            interPointsList = list(itertools.chain.from_iterable(intersectionPointsList))
            for pointInt in interPointsList:
                if pNotIn(pointInt, x, y):
                    flagInter = True

            if flagInter:
                screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation, scaleFactor=scaleFactor,
                                     intersectionPointsList=intersectionPointsList)
                print("\nThe elements of D do not form a packing of R")
                break
        else:
            if animation:
                print("No intersections found at this iteration\n")

        # RIGHT EVENT
        if currentEvent["obj"].typeEvent == "RIGHT":
            deleteElementFromSweepLine(sweepline, int(currentEvent["obj"].circle.name))

        # print sweepline
        if animation:
            if len(sweepline) == 0:
                print("\nThe sweepline is empty!\n")
            else:
                print("Sweepline:")
                pretty(sweepline)

        # inside the rectangle R
        if y.x <= currentEvent["obj"].pointEvent <= y.y and x.x <= currentEvent["obj"].circle.point.y:
            screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation, scaleFactor=scaleFactor)

        count += 1
        if currentEvent["after"] is None:
            print("The event queue is empty!\n")
            print("The elements of D form a packing of R")
            break
        currentEvent = eventsQueue[currentEvent["after"]]  # access to the next event


def isPointInCircle(circle, p, animation, scalefactor=50):
    cx = circle.point.x * scalefactor
    cy = circle.point.y * scalefactor
    px = p.x * scalefactor
    py = p.y * scalefactor
    d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    if animation:
        print(f"\ncircle: {circle}, circleScaled: {cx} {cy}\np Intersection: {p}")
        print(f"Distance point from circle center: {d}, Round(distance): {round(d)}")
        print(f"Circle.radius: {circle.radius}, Circle radius Scaled: {circle.radius * scalefactor}\n")
        print(f"round(d) < circle.radius * scalefactor: {round(d) < circle.radius * scalefactor}\n")

    if round(d) < circle.radius * scalefactor:
        return True
    else:
        return False


def checkIntersection(sweepline, circles, currentEvent, animation):
    for p in sweepline:
        if isPointInCircle(circles[p], currentEvent.point, animation):
            return True

    return False


def pNotIn(point, height, width, scalefactor=1):
    # not add value out the the rectangle
    if point.x > width.y * scalefactor or point.y > height.y * scalefactor:
        return False

    # Negative values are out of the screen, they are not added in the allIntersectionNotChecked
    if point.x < width.x or point.y < height.x:
        return False

    return True


def planeSweepCover(x, y, circles, animation, scaleFactor):
    events = []
    for circle in circles:
        events.append(Event(circle, circle.point.x - circle.radius, "LEFT"))
        events.append(Event(circle, circle.point.x + circle.radius, "RIGHT"))

    print("\nNot ordered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")

    # A way that can be fastest, especially if your list has a lot
    # of records, is to use operator.attrgetter("count")
    try:
        import operator
    except ImportError:
        keyfun = lambda x: x.count  # use a lambda if no operator module
    else:
        keyfun = operator.attrgetter("pointEvent")  # use operator since it's faster than lambda

    events.sort(key=keyfun, reverse=False)  # sort in-place

    print("\nOrdered events:")
    print("name | pointEvent | typeEvent")
    for event in events:
        print(f"{event.name} \t {event.pointEvent} \t {event.typeEvent}")
    print("\n")

    sweepline = {}

    # We set the priority through a dictionary of events
    eventsQueue = {}
    eventsQueue[events[0].name] = {"obj": events[0], "before": None, "after": events[1].name, "point": None}

    for i in range(1, len(events[1::])):
        eventsQueue[events[i].name] = {"obj": events[i], "before": events[i - 1].name, "after": events[i + 1].name,
                                       "point": None}

    eventsQueue[events[-1].name] = {"obj": events[-1], "before": events[-2].name, "after": None, "point": None}

    isThereAnIntersection = False
    count = 0
    countIntersection = 0
    currentEvent = eventsQueue[events[0].name]

    # Swap line among the events
    while True:
        print(f"Iteration: {count}")
        if animation:
            print(f'Name current obj: {currentEvent["obj"].name}')
            print(eventsQueue)
            pretty(eventsQueue)  # pretty view of data
            print(f"Length eventsQueue: {len(eventsQueue)} \n")

        # LEFT EVENT
        if currentEvent["obj"].typeEvent == "LEFT":

            if int(currentEvent["obj"].circle.name) not in sweepline:
                updateSweepLine(sweepline, currentEvent, circles)

        print(f'Type of the event: {currentEvent["obj"].typeEvent}')

        if currentEvent["obj"].typeEvent != "INTERSECTION":
            print(f'Check intersection up and below the circle {currentEvent["obj"].circle.name}')
            for sweepElem in sweepline:
                if sweepElem != int(currentEvent["obj"].circle.name):
                    intersectionPointsList = findIntersection(sweepline, int(currentEvent["obj"].circle.name), circles,
                                                              scaleFactor=1, sweepElem=sweepElem)
                    # screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation, scaleFactor=scaleFactor, intersectionPointsList=intersectionPointsList)

                    if (len(intersectionPointsList[0]) != 0) or (len(intersectionPointsList[1]) != 0):
                        isThereAnIntersection = True

                        if len(intersectionPointsList[0]) == 1:
                            firstP = intersectionPointsList[0]
                            if pNotIn(firstP[0], x, y):
                                cev = currentEvent
                                while True:
                                    if cev["obj"].pointEvent > firstP[0].x:
                                        # Adding a new intersection rectangle event
                                        beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                        objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                        copyObj = deepcopy(cev["obj"])
                                        copyObj.name = objName
                                        copyObj.pointEvent = firstP[1].x
                                        copyObj.point = firstP[1]
                                        copyObj.typeEvent = "INTERSECTION"

                                        eventsQueue[objName] = {
                                            "obj": copyObj,
                                            "after": cev["obj"].name,
                                            "before": beforeTmp
                                        }

                                        eventsQueue[cev["obj"].name]["before"] = objName
                                        eventsQueue[beforeTmp]["after"] = objName
                                        countIntersection += 1
                                        break
                                    else:
                                        cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]

                        elif len(intersectionPointsList[0]) == 2:

                            firstP = intersectionPointsList[0]

                            # if same intersection
                            if firstP[0].x == firstP[1].x and firstP[0].y == firstP[1].y:
                                if pNotIn(firstP[0], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > firstP[0].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = firstP[0].x
                                            copyObj.point = firstP[0]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                            else:
                                if pNotIn(firstP[0], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > firstP[0].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = firstP[0].x
                                            copyObj.point = firstP[0]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                                if pNotIn(firstP[1], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > firstP[1].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = firstP[1].x
                                            copyObj.point = firstP[1]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                        if len(intersectionPointsList[1]) == 1:
                            firstP = intersectionPointsList[1]
                            if pNotIn(firstP[0], x, y):
                                cev = currentEvent
                                while True:
                                    if cev["obj"].pointEvent > firstP[0].x:
                                        # Adding a new intersection rectangle event
                                        beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                        objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                        copyObj = deepcopy(cev["obj"])
                                        copyObj.name = objName
                                        copyObj.pointEvent = thirdP[0].x
                                        copyObj.point = thirdP[0]
                                        copyObj.typeEvent = "INTERSECTION"

                                        eventsQueue[objName] = {
                                            "obj": copyObj,
                                            "after": cev["obj"].name,
                                            "before": beforeTmp
                                        }

                                        eventsQueue[cev["obj"].name]["before"] = objName
                                        eventsQueue[beforeTmp]["after"] = objName
                                        countIntersection += 1
                                        break
                                    else:
                                        cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                        elif len(intersectionPointsList[1]) == 2:
                            thirdP = intersectionPointsList[1]
                            if thirdP[0].x == thirdP[1].x and thirdP[0].y == thirdP[1].y:
                                if pNotIn(thirdP[0], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > thirdP[0].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = thirdP[0].x
                                            copyObj.point = thirdP[0]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                            else:
                                if pNotIn(thirdP[0], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > thirdP[0].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = thirdP[0].x
                                            copyObj.point = thirdP[0]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                                if pNotIn(thirdP[1], x, y):
                                    cev = currentEvent
                                    while True:
                                        if cev["obj"].pointEvent > thirdP[1].x:
                                            # Adding a new intersection rectangle event
                                            beforeTmp = eventsQueue[cev["obj"].name]["before"]
                                            objName = cev["obj"].circle.name + "INTERSECTION" + str(countIntersection)
                                            copyObj = deepcopy(cev["obj"])
                                            copyObj.name = objName
                                            copyObj.pointEvent = thirdP[1].x
                                            copyObj.point = thirdP[1]
                                            copyObj.typeEvent = "INTERSECTION"

                                            eventsQueue[objName] = {
                                                "obj": copyObj,
                                                "after": cev["obj"].name,
                                                "before": beforeTmp
                                            }

                                            eventsQueue[cev["obj"].name]["before"] = objName
                                            eventsQueue[beforeTmp]["after"] = objName
                                            countIntersection += 1
                                            break
                                        else:
                                            cev = eventsQueue[eventsQueue[cev["obj"].name]["after"]]
                    else:
                        if animation:
                            print("No intersections found at this iteration\n")

        # INTERSECTION EVENT
        if currentEvent["obj"].typeEvent == "INTERSECTION":
            if not checkIntersection(sweepline, circles, currentEvent["obj"], animation):
                screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation, scaleFactor=scaleFactor)
                print("The elements of D do not form a cover of R")
                break

        # RIGHT EVENT
        if currentEvent["obj"].typeEvent == "RIGHT":
            deleteElementFromSweepLine(sweepline, int(currentEvent["obj"].circle.name))

        # print sweepline
        if animation:
            if len(sweepline) == 0:
                print("\nThe sweepline is empty!\n")
            else:
                print("Sweepline:")
                pretty(sweepline)

        # inside the rectangle R
        if y.x <= currentEvent["obj"].pointEvent <= y.y and x.x <= currentEvent["obj"].circle.point.y:
            screenShotPlaneSweep(x, y, circles, currentEvent["obj"], animation=animation, scaleFactor=scaleFactor)

        count += 1

        if currentEvent["after"] is None:
            print("The event queue is empty!\n")

            if isThereAnIntersection:
                print("The elements of D form a cover of R")
            else:
                print("The elements of D do not form a cover of R")
            break
        currentEvent = eventsQueue[currentEvent["after"]]  # access to the next event


def bruteForcePacking(x, y, circles, scaleFactor):
    for i in range(len(circles)):
        for j in range(len(circles)):
            intersectionPoints = computeIntersection(circles[i], circles[j], scaleFactor)

            if intersectionPoints is not None:
                print("The elements of D do not form a packing of R")
                return

    print("The elements of D form a packing of R")
    return


def bruteForceCover(x, y, circles, scaleFactor):
    flag = False
    for i in range(len(circles)):
        for j in range(len(circles)):
            intersectionPoints = computeIntersection(circles[i], circles[j], scaleFactor)

            if intersectionPoints is not None:
                for p in intersectionPoints:
                    count = 0
                    for circle in circles:
                        if pNotIn(p, x, y) and isPointInCircle(circle, p, animation=False, scalefactor=50):
                            flag = True
                            pdb.set_trace()
                            break
                        else:
                            count += 1
                    if count == len(circles):
                        pdb.set_trace()
                        print("The elements of D do not form a cover of R")
                        return

    if flag:
        print("The elements of D form a cover of R")
    else:
        print("The elements of D do not form a cover of R")

    return


def main():
    # Read different examples of input
    # file1 = open('input\ExampleInput', 'r')
    # file1 = open('input\ExampleInputTest', 'r')
    # file1 = open('input\ExampleInputTest2', 'r')
    # file1 = open('input\ExampleInputTest3', 'r')
    file1 = open('input\Cover5', 'r')
    # file1 = open('input\Packing6', 'r')
    # file1 = open('input\\100circlesdense', 'r')        # ANIMATION = FALSE
    # file1 = open('input\\1000circlessparse', 'r')      # ANIMATION = FALSE

    # Parse The input
    x, y, circles = parseInput(file1)

    # How big the picture, if activated...
    scaleFactor = 50

    # This is only to see what happens, not plane sweep algorithm applied here
    # draw(x,y,circles, scaleFactor)

    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: ПРОФАЙЛИНГ И ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 80)

    # Запускаем профилирование через отдельный модуль
    profiling.run_comprehensive_profiling(x, y, circles, scaleFactor)

    try:
        sys.exit()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()