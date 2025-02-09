import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector

# Setup OpenCV capture and window size
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detect = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGame:
    def __init__(self, foodPath):
        self.points = []  # Points of the snake
        self.lengths = []  # Distance between points
        self.currentLength = 0  # Total snake length
        self.TotalAllowedLength = 150  # Total allowed length
        self.headPrevious = 0, 0  # Previous head point

        # Food initialization
        self.foodIMG = cv2.imread(foodPath, cv2.IMREAD_UNCHANGED)
        self.foodHeight, self.foodWidth, _ = self.foodIMG.shape
        self.foodLocation = 0, 0
        self.setFoodLocation()  # Initialize food location
        self.score = 0  # Game Score
        self.gameOver = False  # Game Over flag

    def setFoodLocation(self):
        """Set a random location for food"""
        self.foodLocation = random.randint(100, 1000), random.randint(100, 600)

    def update(self, mainIMG, headCurrent):
        """Updates the game state and renders graphics"""
        if self.gameOver:
            cvzone.putTextRect(mainIMG, "Game Over", [250, 350], scale=8, thickness=4, colorT=(255, 255, 255),
                               colorR=(0, 0, 255), offset=20)
            cvzone.putTextRect(mainIMG, f'Your Score: {self.score}', [250, 500], scale=8, thickness=5,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=20)
        else:
            # Get previous and current coordinates
            previousX, previousY = self.headPrevious
            currentX, currentY = headCurrent

            # Add new head position to the snake
            self.points.append([currentX, currentY])
            distance = math.hypot(currentX - previousX, currentY - previousY)
            self.lengths.append(distance)
            self.currentLength += distance
            self.headPrevious = currentX, currentY

            # Limit the snake's length
            if self.currentLength > self.TotalAllowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)

                    if self.currentLength < self.TotalAllowedLength:
                        break

            # Check if snake eats the food
            randomX, randomY = self.foodLocation
            if (
                randomX - self.foodWidth // 2 < currentX < randomX + self.foodWidth // 2
                and randomY - self.foodHeight // 2 < currentY < randomY + self.foodHeight // 2
            ):
                self.setFoodLocation()
                self.TotalAllowedLength += 50
                self.score += 1
                print(f"Score: {self.score}")

            # Draw the snake
            if self.points:
                for i in range(1, len(self.points)):
                    cv2.line(mainIMG, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 255), 20)
                cv2.circle(mainIMG, tuple(self.points[-1]), 20, (200, 0, 200), cv2.FILLED)

            # Collision detection
            if len(self.points) > 10:  # Ensure snake is long enough before checking collision
                pointArray = np.array(self.points[:-2], np.int32).reshape((-1, 1, 2))
                cv2.polylines(mainIMG, [pointArray], False, (0, 200, 0), 3)
                minDist = cv2.pointPolygonTest(pointArray, (currentX, currentY), True)

                if -1 <= minDist <= 1:
                    print("Collision detected!")
                    self.gameOver = True
                    self.points = []
                    self.lengths = []
                    self.currentLength = 0
                    self.TotalAllowedLength = 150
                    self.headPrevious = 0, 0
                    self.setFoodLocation()

            # Draw food
            randomX, randomY = self.foodLocation
            mainIMG = cvzone.overlayPNG(mainIMG, self.foodIMG,
                                        (randomX - self.foodWidth // 2, randomY - self.foodHeight // 2))

            # Display score
            cvzone.putTextRect(mainIMG, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

        return mainIMG


# Initialize the game
game = SnakeGame("Donut.png")
restart_game = False

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    hand, img = detect.findHands(img, flipType=False)

    if hand:
        landmarkList = hand[0]['lmList']
        pointIndex = landmarkList[8][0:2]
        img = game.update(img, pointIndex)
    else:
        img = game.update(img, game.headPrevious)  # Prevent game over when hand is lost

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('r'):
        game.gameOver = False
        game.score = 0
        restart_game = True

    if restart_game:
        game = SnakeGame("Donut.png")
        restart_game = False

    if key == ord('q'):
        break
