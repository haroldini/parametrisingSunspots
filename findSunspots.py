import matplotlib.path as mplp
import skimage.measure as skme
import skimage.io as skio
import os.path as osp
import os
import numpy as np

import plotTools as pt
import inputParams as ip


def sortTheseBy(aList, *bLists):
    return [list(v) for v in zip(*sorted(zip(aList, *bLists), 
                                         key = lambda x: x[0]))]


class Polygon():
    
    index = -1
    register = {}
    sun = None

    @classmethod
    def add(cls, instance):
        cls.register[instance.id] = instance
    
    @classmethod
    def remove(cls, instance):
        del cls.register[instance.id]

    @classmethod
    def _xyFromArray(cls, xy):
        return xy[:, 0], xy[:, 1]
        
    @classmethod
    def _xyFromContour(cls, xy):
        return xy[:, 1], xy[:, 0]
    
    @classmethod
    def getAll(cls):
        return list(cls.register.values())
    
    @classmethod
    def _getSortedByArea(cls, polys):
        areas = [p.getArea() for p in polys]
        areas, polys = sortTheseBy(areas, polys)
        return areas, polys

    @classmethod
    def getSortedByArea(cls):
        return cls._getSortedByArea(cls.getAll())
    

    ''' Classmethods for removing polygons '''
    
    @classmethod
    def removeLargest(cls):
        areas, polys = cls.getSortedByArea()
        maxPoly = polys[len(areas)-1]
        cls.remove(maxPoly)

    @classmethod
    def removeSmallest(cls, minSize):
        areas, polys = cls.getSortedByArea()
        for area, poly in zip(areas, polys):
            if area>minSize: 
                break
            cls.remove(poly)
    
    @classmethod
    def removeEdgy(cls, relativeDiscLimit=0.99): 
        rSquared = (cls.sun.R*relativeDiscLimit)**2
        polys = cls.getAll()
        for poly in polys:
            xz = poly.getPixelCircleCoord()
            if (((xz**2).sum(axis=1) - rSquared) > 0).all():
                cls.remove(poly)           
                    
    @classmethod
    def removeAll(cls):
        polys = cls.getAll()
        for poly in polys:
            cls.remove(poly)
    
    
    ''' Classmethods for parametrising polygons '''
    
    @classmethod
    def _getSignedArea(cls, xy, isArray=True):
        x, y = cls._xyFromArray(xy) if isArray else cls._xyFromContour(xy)
        out = x.dot(np.roll(y, 1))-y.dot(np.roll(x, 1))
        return 0.5*out
    
    @classmethod
    def _getSignedAreaWithError(cls, xy, delta=0.5, isArray=True, \
                                withROCs=False, **kwargs):
        A = cls._getSignedArea(xy, **kwargs)
        x, y = cls._xyFromArray(xy) if isArray else cls._xyFromContour(xy)
        xm, xp = np.roll(x, -1), np.roll(x, +1)
        ym, yp = np.roll(y, -1), np.roll(y, +1)
        dA_dxk = 0.5*(yp - ym) 
        dA_dyk = 0.5*(xm - xp) 
        deltaA = delta*np.sqrt( (dA_dxk**2+dA_dyk**2).sum() )
    
        if withROCs:
            return A, deltaA, dA_dxk, dA_dyk
        return A, deltaA
    
    @classmethod
    def _getArea(cls, xy, **kwargs):
        return abs(cls._getSignedArea(xy, **kwargs))

    @classmethod
    def _getAreaWithError(cls, *args, **kwargs):
        out = list(cls._getSignedAreaWithError(*args, **kwargs))
        out[0] = abs(out[0])      
        return out
        
    @classmethod
    def _getCentroid(cls, xy, isArray=True):
        A = cls._getSignedArea(xy, isArray=isArray)
        x, y = cls._xyFromArray(xy) if isArray else cls._xyFromContour(xy)
        xp, yp = np.roll(x, +1), np.roll(y, +1)
        cx = (x**2).dot(yp  )+(x*xp).dot(yp   )-(x*xp).dot(y   )-(xp**2).dot(y   )
        cy = (x   ).dot(y*yp)+(x   ).dot(yp**2)-(xp  ).dot(y**2)-(xp   ).dot(y*yp)
                
        return np.array([cx, cy])/(6.*A)

    @classmethod
    def _getCentroidWithError(cls, xy, delta=0.5, withROCs=False, isArray=True):
        Cx, Cy = cls._getCentroid(xy, isArray=isArray)
        A, deltaA, dA_dxk, dA_dyk = cls._getSignedAreaWithError(xy, delta=0.5, \
                                                                withROCs=True, \
                                                                isArray=isArray)
        SixA = 6*A
        x, y = cls._xyFromArray(xy) if isArray else cls._xyFromContour(xy)
        xm, xp = np.roll(x, -1), np.roll(x, +1)
        ym, yp = np.roll(y, -1), np.roll(y, +1)
       
        dChi_dxk = (2*x*(yp-ym)) + (xm*(y-ym)) + (xp*(yp-y))
        dChi_dyk = (xm*(xm+x)) - (xp*(x+xp))
        dLam_dxk = (yp*(y+yp)) - (ym*(y+ym))
        dLam_dyk = (xm*(ym+2*y)) - (xm*(yp+2*y)) + (x*(yp-ym))
        dCx_dxk = ( dChi_dxk-6*Cx*dA_dxk ) / SixA
        dCx_dyk = ( dChi_dyk-6*Cx*dA_dyk ) / SixA
        dCy_dxk = ( dLam_dxk-6*Cy*dA_dxk ) / SixA
        dCy_dyk = ( dLam_dyk-6*Cy*dA_dyk ) / SixA
        cROCs = [dCx_dxk, dCx_dyk, dCy_dxk, dCy_dyk]
        
        deltaCx = delta*np.sqrt( (dCx_dxk**2 + dCx_dyk**2).sum() )
        deltaCy = delta*np.sqrt( (dCy_dxk**2 + dCy_dyk**2).sum() ) 
        
        if withROCs:
            return Cx, Cy, deltaCx, deltaCy, cROCs
        return Cx, Cy, deltaCx, deltaCy
    
    @classmethod
    def _getRadius(cls, xy):
        c = cls._getCentroid(xy)
        r = np.sqrt(((xy-c)**2).sum(axis = 1)).mean()
        return r
    
    @classmethod
    def _getRadiusWithError(cls, xy, delta=0.5, withROCs=False, isArray=True):
        x, y = cls._xyFromArray(xy) if isArray else cls._xyFromContour(xy)
        radius = cls._getRadius(xy)
        Cx, Cy, deltaCx, deltaCy, cROCs = cls._getCentroidWithError(xy, withROCs=True)
        dCx_dxk, dCx_dyk, dCy_dxk, dCy_dyk = cROCs[0], cROCs[1], cROCs[2], cROCs[3]
        
        a = x - Cx
        b = y - Cy
        u = a**2 + b**2
        r = np.sqrt(u)
        n = len(r)
        Sai = a/r   
        Sbi = b/r   
        Sa = Sai.mean()
        Sb = Sbi.mean()
        
        dr_dxk = ( Sai/n - dCx_dxk*Sa - dCy_dxk*Sb )
        dr_dyk = ( Sbi/n - dCx_dyk*Sa - dCy_dyk*Sb )
        
        deltaR = delta*np.sqrt( (dr_dxk**2 + dr_dyk**2).sum() ) 
        
        if withROCs:
            return radius, deltaR, dr_dxk, dr_dyk
        return radius, deltaR
    
    @classmethod
    def _getLatitude(cls, poly):
        C = poly.getCentroid() - cls.sun.C
        return np.arcsin(C[1]/cls.sun.R)
    
    
    ''' Instance methods '''
    
    def __init__(self, contour):
        self.xy = np.empty(contour.shape)
        self.xy[:, 0], self.xy[:, 1] = contour[:, 1], contour[:, 0]
        Polygon.index += 1
        self.id = Polygon.index
        self.holes = []
        Polygon.add(self)
    
    def addHole(self, poly):
        self.holes.append(poly)
        
    def getPixelCircleCoord(self):
        return self.xy-Polygon.sun.C
    
    def getSignedArea(self):
        return Polygon._getSignedArea(self.xy)
    
    def getArea(self):
        holeArea = 0
        for hole in self.holes:
            holeArea += hole.getArea()
        return Polygon._getArea(self.xy)-holeArea

    def getAreaWithError(self, withROCs=False):
        
        holeArea = 0
        errorSquared = 0
        for hole in self.holes:
            A, dA = hole.getAreaWithError(withROCs=False)
            holeArea += A
            errorSquared += dA**2
        
        if withROCs:
            A, dA, d1, d2 = Polygon._getAreaWithError(self.xy, withROCs=True)
            errorSquared += dA**2
            return A-holeArea, np.sqrt(errorSquared), d1, d2
        
        A, dA = Polygon._getAreaWithError(self.xy)
        errorSquared += dA**2
        return A-holeArea, np.sqrt(errorSquared)
        
    
    def getCentroid(self):
        return Polygon._getCentroid(self.xy)

    def getCentroidWithError(self, withROCs=False):
        return Polygon._getCentroidWithError(self.xy, withROCs=withROCs)
    
    def getRadius(self):
        return Polygon._getRadius(self.xy)
    
    def getRadiusWithError(self, withROCs=False):
        return Polygon._getRadiusWithError(self.xy, withROCs=withROCs)

    def getPerimeter(self):
        return Polygon._getPerimeter(self.xy)
    
    def getLatitude(self):
        return Polygon._getLatitude(self)
    
    def setAsSun(self):
        x, y = Polygon._xyFromContour(self.xy)    
        self.A, self.deltaA, dA_dxk, \
            dA_dyk = self.getAreaWithError(withROCs=True)
        self.Cx, self.Cy, self.deltaCx, self.deltaCy, \
            cROCs = self.getCentroidWithError(withROCs=True)
        self.dCx_dxk, self.dCx_dyk, self.dCy_dxk, \
            self.dCy_dyk = cROCs[0], cROCs[1], cROCs[2], cROCs[3]
        self.R, self.deltaR, self.dr_dxk, \
            self.dr_dyk = self.getRadiusWithError(withROCs=True)
        self.C = np.array([self.Cx, self.Cy])
        Polygon.sun = self
        

class Umbra(Polygon):

    register = {}

    @classmethod
    def add(cls, instance):
        cls.register[instance.id] = instance
    
    @classmethod
    def remove(cls, instance):
        Polygon.remove(instance)
        del cls.register[instance.id]
    
    @classmethod
    def getAll(cls):
        return list(cls.register.values())
    
    @classmethod
    def isEmpty(cls):
        return len(cls.register.values()) == 0
    
    @classmethod
    def getSortedByArea(cls):
        return cls._getSortedByArea(cls.getAll())   
    
    def __init__(self, *args, **kwargs):
        Polygon.__init__(self, *args, **kwargs)
        Umbra.add(self)


class Penumbra(Polygon):

    register = {}

    @classmethod
    def add(cls, instance):
        cls.register[instance.id] = instance
    
    @classmethod
    def remove(cls, instance):
        Polygon.remove(instance)
        del cls.register[instance.id]
    
    @classmethod
    def getAll(cls):
        return list(cls.register.values())

    @classmethod
    def isEmpty(cls):
        return len(cls.register.values()) == 0

    @classmethod
    def getSortedByArea(cls):
        return cls._getSortedByArea(cls.getAll())  

    @classmethod
    def getHoles(cls):       
        
        polys = cls.getAll()        
        holes = []
        for poly in polys:
            for other in polys:
                if mplp.Path(poly.xy).contains_points(other.xy).all() and poly!=other:
                                       
                    # Check whether the hole lies within any existing hole.
                    subhole = False
                    for hole in holes:
                        if mplp.Path(hole.xy).contains_points(other.xy).all() and hole!=other: 
                            subhole = True
                    
                    # Hole is not counted as hole if it is within an existing hole.
                    # Allows sunspots to exist within a hole within a penumbra.
                    if subhole:
                        continue
                    
                    poly.addHole(other)
                    holes.append(other)
                    Penumbra.remove(other)
                        
    def __init__(self, *args, **kwargs):
        Polygon.__init__(self, *args, **kwargs)
        Penumbra.add(self)


class Sunspot():    
    
    register = {}
    index = -1

    @classmethod
    def add(cls, instance):
        cls.register[instance.id] = instance

    @classmethod
    def getAll(cls):
        return list(cls.register.values())
    
    @classmethod
    def remove(cls, instance):
        del cls.register[instance.id]
        
    @classmethod
    def removeAll(cls):
        sunspots = cls.getAll()
        for sunspot in sunspots:
            cls.remove(sunspot)
            
    @classmethod
    def populate(cls):
        
        # Create a Sunspot object for each penumbra.
        # Store each umbra contained by the Sunspot as a property of the Sunspot.
        umbras = Umbra.getAll()
        centroids = []
        for umbra in umbras:
            centroids.append(Umbra.getCentroid(umbra))
        for penumbra in Penumbra.getAll():
            bools = mplp.Path(penumbra.xy).contains_points(centroids)
            umbraSubset = [u for u, v in zip(umbras, bools) if v]
            cls(penumbra, umbraSubset)
    
    @classmethod
    def countUmbras(cls):
        
        # Return values for the number of umbrae, penumbrae, sunspots.
        uPerP, uPerS = [], []
        uSum, pSum, sSum = 0, 0, 0   
        
        for sunspot in cls.getAll():
            uNum = len(sunspot.umbras)
            uSum += uNum
            pSum += 1
            uPerP.append(uNum)
            if uNum > 0:
                sSum += 1
                uPerS.append(uNum)
                    
        return uSum, pSum, sSum, uPerP, uPerS

    @classmethod
    def getAreaLists(cls):
        
        uAreas, pAreas, sAreas = [], [], []   
        uAreasE, pAreasE, sAreasE = [], [], []   
        uAreasAll, uAreasAllEsqrd = 0., 0.
        pAreasFilteredAll, pAreasFilteredAllEsqrd = 0., 0.
        
        for sunspot in cls.getAll():
            us, usE, p, pE, ua, uaE = sunspot.getAreas()
            uAreas += us
            uAreasE += usE
            pAreas.append(p)
            pAreasE.append(pE)
            sAreas.append(p+ua)
            sAreasE.append(np.sqrt(pE**2+uaE**2))
            uAreasAll += ua
            uAreasAllEsqrd += uaE
            
            if ua != 0:              
                pAreasFilteredAll += p
                pAreasFilteredAllEsqrd += pE**2
        
        pAreasAll = sum(pAreas)
        pAreasAllE = np.sqrt(sum([E**2 for E in pAreasE]))
        uAreasAllE = np.sqrt(uAreasAllEsqrd)
        
        if pAreasFilteredAll>0:
            upRatioTot = uAreasAll/pAreasFilteredAll
            upRatioTotE = (pAreasFilteredAllEsqrd/(pAreasFilteredAll**2)) + \
            np.sqrt((uAreasAll**2/(pAreasFilteredAll**4))*pAreasFilteredAllEsqrd)
        else:
            upRatioTot = np.NaN
            upRatioTotE = np.NaN
        
        return [uAreas, uAreasE], [pAreas, pAreasE], [sAreas, sAreasE], \
               [uAreasAll, uAreasAllE], [pAreasAll, pAreasAllE], \
               [upRatioTot, upRatioTotE], \
               [pAreasFilteredAll, np.sqrt(pAreasFilteredAllEsqrd)]
               
                 
    def __init__(self, penumbra, umbras):
        self.umbras = umbras
        self.penumbra = penumbra
        Sunspot.index += 1
        self.id = Sunspot.index
        Sunspot.add(self)
        
    def getAreas(self):
        pArea, pAreaE = self.penumbra.getAreaWithError()
        uAreas, uAreasE = [], []
        for umbra in self.umbras:
            uArea, uAreaE = umbra.getAreaWithError()
            uAreas.append(uArea)
            uAreasE.append(uAreaE)
        
        ua = sum(uAreas)
        uaE = sum([E**2 for E in uAreasE])
        pArea -= ua
        pAreaE += np.sqrt(uaE**2)
        
        return uAreas, uAreasE, pArea, pAreaE, ua, uaE
    

def getSun(im):
    
    # Sun = largest polygon from contours generated with level=0.05
    threshold = 0.05
    areas = []
    polys = skme.find_contours(im, level = threshold, fully_connected = 'high')
    for poly in polys:       
        areas.append(Polygon._getArea(poly, isArray=False))
        
    areas, polys = sortTheseBy(areas, polys)
    sunPoly = Polygon(polys[-1])
    sunPoly.setAsSun()
    
    return sunPoly


def limbCorrection(im, sunRadius, sunCentre):
    u = ip.u
    x, y = np.ogrid[:im.shape[0], :im.shape[1]]
    distFromCentre = np.sqrt((x-sunCentre[0])**2 + (y-sunCentre[1])**2)
    limbCorrection = 1-u*(1-np.sqrt((sunRadius**2-distFromCentre**2)/sunRadius**2))
    im /= limbCorrection  
    return im


def getImgData(filepath, numImgs):
    
    # Identify the Sun polygon.
    im = skio.imread(filepath, as_gray=True)
    sun = getSun(im)
    sunCentre = sun.getCentroid()
    sunRadius = sun.getRadius()
    imgData = {"sunRadius":sunRadius, "sunCentre":sunCentre}
    
    # Correct limb darkening.
    im = limbCorrection(im, sunRadius, sunCentre)
    if output["plotLimbCorrection"] == True:
        imgData.update({"im":im})
    imgData.update({"filepath":filepath})

    # Get contours for umbrae and penumbrae.
    pLevel = ip.pThreshold
    uLevel = ip.uThreshold
    for contour in skme.find_contours(im, level=pLevel, fully_connected='high'):
        Penumbra(contour)
    for contour in skme.find_contours(im, level=uLevel, fully_connected='high'):
        Umbra(contour)
    Penumbra.removeLargest()
    Umbra.removeLargest()
    
    # Remove anomalous contours.
    if not output["plotAnomalies"]:
        pMinSize = ip.pMinSize
        Penumbra.removeSmallest(pMinSize)
        Penumbra.removeEdgy()
        uMinSize = ip.uMinSize
        Umbra.removeSmallest(uMinSize)
        Umbra.removeEdgy()  
    Penumbra.getHoles()
    
    # If umbrae or penumbrae are not present, return NaN data.
    if Umbra.isEmpty() or Penumbra.isEmpty():
        imgData.update({"uSum":0, "pSum":0, "sSum":0,
                        "uLats":[], "pLats":[],
                        "tus":[], "tps":[],
                        "uPerP":np.NaN, "uPerS":np.NaN,
                        "uPerPAvg":np.NaN, "uPerSAvg":np.NaN,
                        "uAreas":[[np.NaN], [np.NaN]], 
                        "uAreasAll":[np.NaN, np.NaN],
                        "pAreas":[[np.NaN], [np.NaN]], 
                        "pAreasAll":[np.NaN, np.NaN],      
                        "upRatioTot":[np.NaN, np.NaN], 
                        "pAreasFilteredAll":[np.NaN, np.NaN],
                        "contours":[]
                        })
        return imgData  
    
    # Store contours if they are to be plotted.
    if output["plotContours"] or output["plotAnimate"] or output["plotCentroids"]:
        contours = []
        for contour in [Penumbra, Umbra]:
            areas, polys = contour.getSortedByArea()        #areas, polys
            points = []                                     #centroids
            for poly in contour.getAll():
                points.append(poly.getCentroid())
            
            contour = {"areas":areas, "polys":polys, "points":points}
            contours.append(contour)
        imgData.update({"contours":contours})

    # Populate sunspot class & get sunspot properties.
    Sunspot.populate()
    uSum, pSum, sSum, uPerP, uPerS  = Sunspot.countUmbras() 
    uLats = [u.getLatitude() for u in Umbra.getAll()]
    pLats = [p.getLatitude() for p in Penumbra.getAll()]
    tus = [filepath for u in Umbra.getAll()]
    tps = [filepath for p in Penumbra.getAll()]
    
    if len(uPerS) == 0 or len(uPerP) == 0:
        imgData.update({"uPerP":[], "uPerS":[],
                        "uPerPAvg":np.NaN, "uPerSAvg":np.NaN})
    else:
        uPerPAvg = sum(uPerP)/len(uPerP)
        uPerSAvg = sum(uPerS)/len(uPerS)
        imgData.update({"uPerP":uPerP, "uPerS":uPerS,
                        "uPerPAvg":uPerPAvg, "uPerSAvg":uPerSAvg})

    uAreas, pAreas, sAreas, uAreasAll, pAreasAll, upRatioTot, pAreasFilteredAll = \
        Sunspot.getAreaLists()
    imgData.update({"uSum":uSum, "pSum":pSum, "sSum":sSum,
                     "uLats":uLats, "pLats":pLats,
                     "tus":tus, "tps":tps,
                     "uAreas":uAreas, "uAreasAll":uAreasAll,
                     "pAreas":pAreas, "pAreasAll":pAreasAll,   
                     "sAreas":sAreas,
                     "upRatioTot":upRatioTot, 
                     "pAreasFilteredAll":pAreasFilteredAll
                     })
    
    return imgData


def getAllImgData(foldername):

    allImgData = []  
    
    # Perform sunspot detection on sequence of images in path.
    filelist = os.listdir(foldername)
    for i, filename in enumerate(filelist):
        print(i, "/", len(filelist), "|", filename)
        filepath = osp.join(foldername, filename)
        numImgs = len(filelist)
        imgData = getImgData(filepath, numImgs)  
        
        # Empty registers for next image.
        Umbra.removeAll()
        Penumbra.removeAll()
        Sunspot.removeAll()
        Polygon.removeAll()
        Polygon.sun = None
        
        # Store data from each image as a list of dictionaries.
        allImgData.append(imgData)

    return allImgData


def main(): 
    
    # Name of folder which contains images to be analysed.
    foldername = "09-11"                  
    allImgData = getAllImgData(foldername) 
    pt.main(allImgData, output, foldername)
        
output = ip.output
main()

