class Polygon():    
    
    # Return the polygons sorted by their perimeter length.
    @classmethod
    def _getSortedByPerimeter(cls, polys):
        areas = [p.getArea() for p in polys]
        perimeters = [p.getPerimeter() for p in polys]
        perimeters, areas, polys = sortTheseBy(perimeters, areas, polys)
        return perimeters, areas, polys

    @classmethod
    def getSortedByPerimeter(cls):
        return cls._getSortedByPerimeter(cls.getAll())
    
    # Remove polygons below a minimum roundness value.
    @classmethod
    def removeIrregular(cls, minRoundnessSqrd): 
        perimeters, areas, polys = cls.getSortedByPerimeter()
        for perimeter, area, poly in zip(perimeters, areas, polys):
            roundnessSqrd = 4*np.pi*area/perimeter**2
            if roundnessSqrd < minRoundnessSqrd:
                cls.remove(poly)  
                
    
    @classmethod
    def _getPerimeter(cls, xy):
        p = 0
        for i in range(len(xy)-1):            
            p += np.linalg.norm(xy[i]-xy[i+1])
            if i+2 == len(xy):
                p += np.linalg.norm(xy[-1]-xy[0])
        return p
    
    @classmethod
    def _getLatitudes(cls, poly):
        xz = poly.getPixelCircleCoord()
        return np.arcsin(xz[:,1]/cls.sun.R)
    
    # Calculate latitudes with error.
    # Unused, as latitude analysis consisted of butterfly diagrams only, so error was not needed.
    @classmethod
    def _getLatitudesWithError(cls, xy, delta=0.5):
        lat = cls._getLatitudes(xy)
        rSun = cls.sun.R
        Cy = cls.sun.Cy
        y2 = xy[:,1] - Cy
        drSun_dxk, drSun_dyk = cls.sun.dr_dxk, cls.sun.dr_dyk
        dCy_dxk, dCy_dyk = cls.sun.dCy_dxk, cls.sun.dCy_dyk
        
        #Error on latitude.
        deltaLat = np.empty(y2.shape)
        dlat_dyi = 1/np.sqrt(rSun**2 - y2**2)
        for i in range(len(y2)):
            dlat_dxk = (-dCy_dxk-(y2[i]/rSun)*drSun_dxk) / np.sqrt(rSun**2 - y2[i]**2)
            dlat_dyk = (-dCy_dyk-(y2[i]/rSun)*drSun_dyk) / np.sqrt(rSun**2 - y2[i]**2)
            deltaLat[i] = delta*np.sqrt( (dlat_dxk**2 + dlat_dyk**2).sum() + dlat_dyi[i]**2)
        
        return lat, deltaLat
    
    # Return the depth of a coordinate.
    @classmethod
    def _getZ(cls, xy, delta=0.5):
        x, y = xy[:,0], xy[:,1]
        Cx, Cy = cls.sun.Cx, cls.sun.Cy
        rSun = cls.sun.R
        drSun_dxk, drSun_dyk = cls.sun.dr_dxk, cls.sun.dr_dyk
        dCx_dxk, dCx_dyk = cls.sun.dCx_dxk, cls.sun.dCx_dyk
        dCy_dxk, dCy_dyk = cls.sun.dCy_dxk, cls.sun.dCy_dyk
        deltaCx, deltaCy = cls.sun.deltaCx, cls.sun.deltaCy
        
        x2 = x - Cx
        y2 = y - Cy
        z2 = np.sqrt(rSun**2 - x2**2 - y2**2)

        # Calculate error in z value.
        dz2_dxi = -x2/z2
        dz2_dyi = -y2/z2
        delta_x2 = np.sqrt(delta**2 + deltaCx**2)
        delta_y2 = np.sqrt(delta**2 + deltaCy**2)
        delta_z2 = np.empty(z2.shape)
        for i in range(len(x)):
            dz2_dxk = ( rSun*drSun_dxk + x2[i]*dCx_dxk + y2[i]*dCy_dxk )/z2[i]
            dz2_dyk = ( rSun*drSun_dyk + x2[i]*dCx_dyk + y2[i]*dCy_dyk )/z2[i]
            delta_z2[i] = delta*np.sqrt( (dz2_dxk**2 + dz2_dyk**2).sum() 
                                        + dz2_dxi[i]**2 + dz2_dyi[i]**2 )
        
        return [x2, z2, y2], [delta_x2, delta_z2, delta_y2]

    # Convert coordinates to 3D sphere coordinates.
    def getPixelSphereCoord(self):
        xz = self.getPixelCircleCoord()
        x, z = xz[:,0], xz[:,1]
        y = np.sqrt(Polygon.sun.R**2 - x**2 - z**2)
        return x, y, z

    def getLatitudes(self):
        return Polygon._getLatitudes(self)