u = 0.55

pThreshold = 0.75
pMinSize = 25
pMinRoundness = 0.05
uThreshold = 0.40
uMinSize = 16
uMinRoundness = 0.2

rSunM = 6.957*(10**8)   #Radius of sun in metres https://en.wikipedia.org/wiki/Solar_radius

output = {"plotVisual":True, 
          "plotButterfly":False, 
          "plotAreaDistribution":False, 
          "plotLatDistribution":False,
          "printDataTable":False,
          
          #Export settings
          "plotAnimate":True,
          "exportAnim":"mp4",                         #gif or mp4 (or none)
          "exportCSV":False,
          "exportPlots":True,
           
          #Plot settings
          "plotCentroids":False,
          "plotContours":True, 
          "plotLimbCorrection":False, 
          "plotStats":False,
          "plotSun":False,

          
          #Visual settings
          "plotLim":None, #formation((1000, 2000), (2000, 2500)),            #((1500,1800),(2500,2200)),      #((xlim1, xlim2),(ylim1, ylim2)) or None
          "colorContourBySize":False, 
          "fillContours":True,
          "plotInverse":True, 
          "plotAnomalies":False,
          "fps":30, 
          "sf":4}     

