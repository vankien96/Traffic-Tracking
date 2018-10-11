from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=10):
        # initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.initLocation = OrderedDict()
        self.currentLocation = OrderedDict()
    
    #Register new object
    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.currentLocation[self.nextObjectID] = rect
        self.nextObjectID += 1
    
    #Remove object
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.initLocation.keys():
            del self.initLocation[objectID]
        del self.currentLocation[objectID]
    
    #accepts a list of bounding box rectangles, 
    def update(self, rects):
        if len(rects) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
    
            return self.objects
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        # otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object centroids
        else:
            objectIDs = list(self.objects.keys())
            objectCentroid = list(self.objects.values())

            # compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing object centroid
            dist = distance.cdist(np.array(objectCentroid), inputCentroids)
            
            # in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index list
            rows = dist.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
            cols = dist.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
				# column value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue
                
                # otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.currentLocation[objectID] = rects[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, dist.shape[0])).difference(usedRows)
            unusedCols = set(range(0, dist.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if dist.shape[0] >= dist.shape[1]:
                for row in unusedRows:
                    # grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    # check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # otherwise, if the number of input centroids is greater
			    # than the number of existing object centroids we need to
			    # register each new input centroid as a trackable object
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.objects