import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.animation as anim
import matplotlib.cm as mplc
import matplotlib.lines as lines
from prettytable import PrettyTable as prt

from colorsys import hsv_to_rgb
import skimage.io as skio
import numpy as np
import datetime
import os
import os.path as osp

import inputParams as ip

output = ip.output

mpl.rcParams['animation.ffmpeg_path'] = r'Z:\Libraries\ffmpeg\ffmpeg-2022-03-24-git-28d011516b-essentials_build\ffmpeg-2022-03-24-git-28d011516b-essentials_build\bin\ffmpeg.exe'

# Defining colours to use across plots.
colorBG = 'white'
colorFG = 'black'
colorU = "#70139e"
colorP = "#139438"
colorP2 = "#15388f"
colors = [colorP, colorU]

if output["plotInverse"]==True:
    colorBG, colorFG = colorFG, colorBG

labelkwargs = {'ha':'center', 'size':13, 'color':colorFG, \
               'fontname':'Open Sans'}
titlekwargs = {**labelkwargs, "fontweight":500, "size":15}
mathkwargs = {**labelkwargs, 'style':'italic'}

if not output["plotLimbCorrection"]:
    titlekwargs["color"] = colorBG



def savePlot(filename, foldername):
    outFolder = foldername+" Output"
    path = osp.join(os.getcwd(), outFolder)
    
    if not osp.exists(path):
        os.makedirs(path)
    newfilename = osp.join(path, filename)
    return newfilename


def pixToM(pixels, radius, area=True):
    metresPerPixel = ip.rSunM/radius
    metres = pixels*metresPerPixel
    if area:
        metres*=metresPerPixel
    return metres


def fileToDatetime(filepath, foldername, universal=False):
    date = filepath[len(foldername)+1:-15]
    y = int(date[:-12])
    mo = int(date[5:-9])
    d = int(date[8:-6])
    h = int(date[11:-3])
    mi = int(date[14:])

    t = datetime.datetime(y, mo, d, h, mi, 00)
    if universal:
        t = (t-datetime.datetime(1970,1,1)).total_seconds()
    return t


def fileToDateString(filepath, foldername):
    date = filepath[len(foldername)+1:-15]
    date = date.replace("_", "/", 2)
    date = date.replace("_", " | ", 1)
    date = date.replace("_", ":", 1)
    return date
    

def formatAxes(axes):
    for ax in axes:
        ax.set_facecolor(colorBG)
        ax.spines['bottom'].set_color(colorFG)
        ax.spines['left'].set_color(colorFG)
        ax.spines['top'].set_color(colorFG)
        ax.spines['right'].set_color(colorFG)
        ax.tick_params(axis='x', colors=colorFG)
        ax.tick_params(axis='y', colors=colorFG)
    return axes


#Formats each axis.
def clearAxes(axes, radius, centre, output, clear=False):
    for ax in axes:
        if output["plotAnimate"] and clear:
            ax.clear()   
        formatAx(ax)
        limitAx(ax, radius, centre)
    return axes


def limitAx(ax, radius, centre):
    if output["plotLim"] != None:
        ax.set_xlim(output["plotLim"][0])
        ax.set_ylim(output["plotLim"][1])
        pass
    else:
        ax.set_xlim(centre[0]-1.1*radius, centre[0]+1.1*radius)
        ax.set_ylim(centre[1]-1.1*radius, centre[1]+1.1*radius)
    
    
def formatAx(ax):
    ax.set_facecolor(colorBG)
    ax.spines['bottom'].set_color(colorBG)
    ax.spines['left'].set_color(colorBG)
    ax.spines['top'].set_color(colorBG)
    ax.spines['right'].set_color(colorBG)
    ax.tick_params(axis='x', colors=colorBG)
    ax.tick_params(axis='y', colors=colorBG)
    
    
def imMask(im, radius, centre, cut=1.05):
    x, y = np.ogrid[:im.shape[0], :im.shape[1]]
    distFromCentre = np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    im = np.ma.masked_where(distFromCentre > radius*cut, im)
    return im
    

def plotContours(contours, ax, output, update=False):
    updateList = []
    hues = [0.5, 0]
    
    for i, contour in enumerate(contours):
        areas, polys, points = contour["areas"], contour["polys"], \
            contour["points"]
        maxArea = max(areas)
        xs, ys = zip(*points)
    
        hue = hues[i]
        colorVar = colors[i]
        
        if output["plotContours"]:
            for area, poly in zip(areas, polys):
                if output["colorContourBySize"]: 
                    hue = hues[i] + area/(3*maxArea)
                    colorVar = hsv_to_rgb(hue, 1., 1.)
                
                if output["fillContours"] == False:
                    polyPlot = ax.plot(poly.xy[:, 0], poly.xy[:, 1],
                                 c = colorVar, lw=1, zorder=2)
                else:
                    polyPlot = ax.plot(poly.xy[:, 0], poly.xy[:, 1],
                                  c = "k", lw=0, zorder=2)
                    polyFill = ax.fill(poly.xy[:, 0], poly.xy[:, 1],
                                 c = colorVar, zorder=1)
                    if update: updateList.append(polyFill[0])
                if update: updateList.append(polyPlot[0])
            
        if output["plotCentroids"]:
            if output["fillContours"] and output["plotContours"]:
                pointPlot = ax.plot(xs, ys, '.', ms=20, c = 'k', zorder=3)
            else:
                pointPlot = ax.plot(xs, ys, '.', ms=20, c = colorVar, zorder=3)
            if update: updateList.append(pointPlot[0])
        
    return updateList
                            


def plotVisual(allImgData, output, foldername):

    # Determines number of plots / frames. Creates multiple static plots if 
    # animation disabled, or 1 plot with many frames if animation enabled.
    numPlots = len(allImgData)
    if output["plotAnimate"]: numPlots, numFrames = 1, numPlots
            
    for i in range(numPlots):
        imgData = allImgData[i]
        radius, centre = imgData["sunRadius"], imgData["sunCentre"]
        fig = plt.figure(facecolor = colorBG)
        
        if output["plotLimbCorrection"]:
            fig.set_size_inches(10, 8, forward=True)
        else:
            fig.set_size_inches(8, 8, forward=True)
                
        #Grid0
        if output["plotStats"]:
            grid0 = gs.GridSpec(2, 1, height_ratios=[2, 1])
        else:
            grid0 = gs.GridSpec(1, 1)

        # Subgrid for grid0[0]
        if output["plotLimbCorrection"] and output["plotSun"]:
            grid00 = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid0[0], \
                                                width_ratios=[2, 1], \
                                                height_ratios=[1, 1])
        else:
            grid00 = gs.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid0[0])
                        
        # Primary plot.
        ax1 = fig.add_subplot(grid00[0:, 0])
        ax1.set_title("Source Image", titlekwargs)
        axes = [ax1]
        
        if output["plotSun"]:
            imSun = skio.imread(imgData["filepath"], as_gray=True)
                            
            if output["plotInverse"] == False:
                cut = 1.05
            else:
                cut = 1.00
            
            imSun = imMask(imSun, radius, centre, cut=cut)
            imSunAx1 = ax1.imshow(imSun, cmap = mplc.gray)

            # Show extra plots for limb correction and original image.
            if output["plotLimbCorrection"]: 
                imCorrect = imgData["im"]
                if output["plotInverse"]:
                    imCorrect = imMask(imCorrect, radius, centre, cut=1.00)
                ax2 = fig.add_subplot(grid00[0, 1], sharex=ax1, sharey=ax1)
                ax3 = fig.add_subplot(grid00[1, 1], sharex=ax1, sharey=ax1)
                ax2.set_title("Source Image", titlekwargs)
                ax3.set_title("Limb Darkening Corrected Image", titlekwargs)
                imSunAx2 = ax2.imshow(imSun, cmap = mplc.gray)
                imCorrectAx3 = ax3.imshow(imCorrect, cmap = mplc.gray)
                axes.extend([ax2, ax3])
        
        # If sun not shown, plot circle.
        else:
            sunCircle = plt.Circle((imgData["sunCentre"]), imgData["sunRadius"], \
                                   color = "#3d3d3d")
            sunCirclePlot = ax1.add_patch(sunCircle)
            plt.gca().set_aspect('equal')

        # Show stats of image.
        if output["plotStats"]:
            
            ax4 = fig.add_subplot(grid0[1, 0])
            ax4.set_facecolor(colorBG)

            uSum, pSum = imgData["uSum"], imgData["pSum"]
            uPerPAvg, uPerSAvg = imgData["uPerPAvg"], imgData["uPerSAvg"]
            uAreasAll, pAreasAll = imgData["uAreasAll"], imgData["pAreasAll"]
            upRatioTot = imgData["upRatioTot"]
            date = fileToDateString(imgData["filepath"], foldername)
            fileToDatetime(imgData["filepath"], foldername, universal=True)
            
            textDate = fig.text(0.15, 0.35, f"{date}", titlekwargs, ha = "left")
            textCounter  = fig.text(0.15, 0.32, f"Image {i+1} of {len(allImgData)}", labelkwargs, ha="left")
            fig.add_artist(lines.Line2D([0.1, 0.9], [0.3, 0.3], linewidth=2, color=colorFG))
            text_uSum = fig.text(0.15, 0.25, f"Number of Umbrae = {uSum}", labelkwargs, ha="left")
            text_pSum = fig.text(0.15, 0.22, f"Number of Penumbrae = {pSum}", labelkwargs, ha="left")
            text_uPerPAvg = fig.text(0.15,0.19, f"Average Umbrae per Penumbra = {round(uPerPAvg, 3)}", labelkwargs, ha="left")
            text_uPerSAvg = fig.text(0.15,0.16, f"Average Umbrae per Sunspot = {round(uPerSAvg, 3)}", labelkwargs, ha="left")
            text_uAreas = fig.text(0.15,0.13, f"Umbral Area (Total) = {round(uAreasAll[0], 2)}"+r" $\pm$ "+f"{round(uAreasAll[1], 3)}", labelkwargs, ha="left")
            text_pAreas = fig.text(0.15,0.10, f"Penumbral Area (Total) = {round(pAreasAll[0], 2)}"+r" $\pm$ "+f"{round(pAreasAll[1], 3)}", labelkwargs, ha="left")
            text_upRatioTot = fig.text(0.15,0.07, f"Umbral/Penumbral Area Ratio = {round(upRatioTot[0], 3)}"+r" $\pm$ "+f"{round(upRatioTot[1], 4)}", labelkwargs, ha="left")
            
            formatAx(ax4)  
            limitAx(ax4, radius, centre)             
                            
        # Formatting each axis
        axes = clearAxes(axes, radius, centre, output, clear=True)
            
        # Show sunspot centroids & polygons
        if output["plotContours"] or output["plotCentroids"]:
            contours = imgData["contours"]
            plotContours(contours, ax1, output)
    
        if output["plotAnimate"]:
                                                         
            # Animation update function
            def animate(i):
                updateList = []             # List of objects to update next frame.
                imgData = allImgData[i]     # Data for next frame from dict.
                
                #Reformat axes for new frame
                radius, centre = imgData["sunRadius"], imgData["sunCentre"]
                axes2 = [ax1]
                ax1.clear()
                                    
                if output["plotStats"]:
                    uSum, pSum = imgData["uSum"], imgData["pSum"]
                    uPerPAvg, uPerSAvg = imgData["uPerPAvg"], imgData["uPerSAvg"]
                    uAreasAll, pAreasAll = imgData["uAreasAll"], imgData["pAreasAll"]
                    upRatioTot = imgData["upRatioTot"]
                    date = fileToDateString(imgData["filepath"], foldername)
                    textDate.set_text(f"{date}")
                    textCounter.set_text(f"Image {i+1} of {len(allImgData)}")
                    text_uSum.set_text(f"Number of Umbrae = {uSum}")
                    text_pSum.set_text(f"Number of Penumbrae = {pSum}")
                    text_uPerPAvg.set_text(f"Average Umbrae per Penumbra = {round(uPerPAvg, 3)}")
                    text_uPerSAvg.set_text(f"Average Umbrae per Sunspot = {round(uPerSAvg, 3)}")
                    text_uAreas.set_text(f"Umbral Area (Total) = {round(uAreasAll[0], 2)}"+r" $\pm$ "+f"{round(uAreasAll[1], 3)}")
                    text_pAreas.set_text(f"Penumbral Area (Total) = {round(pAreasAll[0], 2)}"+r" $\pm$ "+f"{round(pAreasAll[1], 3)}")
                    text_upRatioTot.set_text(f"Umbral/Penumbral Area Ratio = {round(upRatioTot[0], 3)}"+r" $\pm$ "+f"{round(upRatioTot[1], 4)}")
            
                #Update sun image.
                if output["plotSun"]:
                    imSun = skio.imread(imgData["filepath"], as_gray=True)
                    if output["plotInverse"] == False:
                        imSun = imMask(imSun, radius, centre)
                    imSunAx1.set_data(imSun)
                    ax1.imshow(imSun, cmap = mplc.gray)
                    ax1.set_title("Source Image With Sunspot Overlay", \
                                  titlekwargs)
                    updateList.append(imSunAx1)
                    
                    #Update colour correction image.
                    if output["plotLimbCorrection"]:
                        ax2.clear()
                        ax3.clear()
                        imSunAx2.set_data(imSun)
                        imCorrect = imgData["im"]
                        imCorrectAx3.set_data(imCorrect)
                        updateList.append(imSunAx2)
                        updateList.append(imCorrectAx3)
                        ax2.imshow(imSun, cmap = mplc.gray)
                        ax3.imshow(imCorrect, cmap = mplc.gray)
                        ax2.set_title("Source Image", titlekwargs)
                        ax3.set_title("Limb Darkening Corrected Image", \
                                      titlekwargs)
                        axes2.extend([ax2, ax3])

                #Update sun image with new radius / centroid if sun disabled.
                else:
                    sunCirclePlot = ax1.add_patch(sunCircle)
                    updateList.append(sunCirclePlot)

                if output["plotContours"] or output["plotCentroids"]:
                    contours = imgData["contours"]
                    updateList.extend(plotContours(contours, ax1, output,\
                                                   update=True))
                                    
                axes = clearAxes(axes2, radius, centre, output, clear=False)       
                fig.canvas.draw()
                return updateList

            fps = output["fps"]
            ani = anim.FuncAnimation(fig, animate, frames=numFrames, \
                                     interval=1000 / fps, blit=True, repeat=True)
            
            # Save plot.
            if output["exportAnim"] == "mp4":
                filename = savePlot("Animation.mp4", foldername)
                ani.save(filename, dpi=300, writer=anim.FFMpegWriter(fps=fps, \
                         bitrate=1000) , savefig_kwargs=dict(facecolor=colorBG))
            elif output["exportAnim"] == "gif":
                filename = savePlot("Animation.gif", foldername)
                ani.save(filename, writer=anim.PillowWriter(fps=fps), \
                         savefig_kwargs=dict(facecolor=colorBG))
                
        else:
            # Save plot.
            if output["exportPlots"]:
                datestring = imgData["filepath"][len(foldername)+1:-15]
                filename = savePlot(f"Visual {datestring}.png", foldername)
                plt.savefig(filename, dpi=500, facecolor=colorBG)


def plotButterfly(allImgData, foldername):
    
    # Initialise figure.
    fig = plt.figure(figsize = (12, 18), facecolor=colorBG)
    grid = gs.GridSpec(4, 1)
    ax00 = fig.add_subplot(grid[0])
    ax10 = fig.add_subplot(grid[1])
    ax20 = fig.add_subplot(grid[2])
    ax30 = fig.add_subplot(grid[3])
    
    # Unpacking relevant data.   
    uNum, pNum, sNum = [], [], []
    uAreasSum, uAreasSumE = [], []
    pAreasSum, pAreasSumE = [], []
    upRatioTot, upRatioTotE = [], []
    pAreasFilteredSum, pAreasFilteredSumE = [], []
    radii = []
    t = []
    tu, uLats = [], []
    tp, pLats = [], []

    for imgData in allImgData:
        
        # Data for sunspot number graph.
        uNum.append(imgData["uSum"])
        pNum.append(imgData["pSum"])
        sNum.append(imgData["sSum"])
    
         #Data for area & area ratio graph.
        uAreasSum.append(imgData["uAreasAll"][0])
        pAreasSum.append(imgData["pAreasAll"][0])
        uAreasSumE.append(imgData["uAreasAll"][1])
        pAreasSumE.append(imgData["pAreasAll"][1])
        upRatioTot.append(imgData["upRatioTot"][0])
        upRatioTotE.append(imgData["upRatioTot"][1])
        pAreasFilteredSum.append(imgData["pAreasFilteredAll"][0])
        pAreasFilteredSumE.append(imgData["pAreasFilteredAll"][1])
        
        # Data for all plots.
        ti = fileToDatetime(imgData["filepath"], foldername)
        t.append(ti)
        radii.append(imgData["sunRadius"])
        
        # Data for butterfly diagram
        uLats.extend(imgData["uLats"])
        pLats.extend(imgData["pLats"])
        tui = [fileToDatetime(filepath, foldername) for filepath in imgData["tus"]]
        tpi = [fileToDatetime(filepath, foldername) for filepath in imgData["tps"]]
        tu.extend(tui)
        tp.extend(tpi)
    
    # Converting units of data before plot.
    uLats, pLats = [uLat*90 for uLat in uLats], [pLat*90 for pLat in pLats]
    for i, radius in enumerate(radii):
        pAreasSum[i] = pixToM(pAreasSum[i], radius, area=True)
        pAreasSumE[i] = pixToM(pAreasSumE[i], radius, area=True)
        pAreasFilteredSum[i] = pixToM(pAreasFilteredSum[i], radius, area=True)
        pAreasFilteredSumE[i] = pixToM(pAreasFilteredSumE[i], radius, area=True)
        uAreasSum[i] = pixToM(uAreasSum[i], radius, area=True)
        uAreasSumE[i] = pixToM(uAreasSumE[i], radius, area=True)
    
    
    # Plotting params.
    maWidth = 28
    maString = "4-Week"
    dateString = "During Solar Cycle 24"
    bfSize = 4                      #4 for solar cycle 24     #33 for ar12192
    bfAlpha = 0.25                  #0.25 for solar cycle 24    #0.25 for ar12192
    scSize = 1
    lw = 2
    bfYLim = [-60, 60]
    scYLim = [0, max(uAreasSum+pAreasFilteredSum)]
    upYLim = [0, min(max(upRatioTot)+0.05, 0.4)]


    # Moving average function.
    def movingAvg(x, width):
        window = np.ones(int(width))/float(width)
        return np.convolve(x, window, 'same')

    
    # Butterfly diagram using scatter plot.
    ax00.scatter(tp+tu, pLats+uLats, label="Sunspot Density", marker=",", \
                 s=bfSize, alpha=bfAlpha, color=colorFG, edgecolor="none")
    ax00.set_title(f"Density of Detected Sunspots on the Visible Hemisphere of the Sun by Latitude {dateString}.", \
                   labelkwargs, y=1.05, color=colorFG)
    ax00.set_ylabel(r'Latitude ($^\circ$)', labelkwargs, labelpad=10, \
                    color=colorFG)
    ax00.set_ylim(bfYLim) 


    # Umbral & Penumbral area plot
    ax10.errorbar(t, pAreasFilteredSum, yerr=pAreasFilteredSumE, fmt="o", \
                  ms=scSize, label="Penumbral Area", color=colorP)
    ax10.errorbar(t, uAreasSum, yerr=uAreasSumE, fmt="o", ms=scSize, \
                  label="Umbral Area", color=colorU)
    ax10.plot(t, movingAvg(pAreasFilteredSum, maWidth), lw=lw, color=colorP)
    ax10.plot(t, movingAvg(uAreasSum, maWidth), lw=lw, color=colorU)
    ax10.set_title(f"{maString} Moving Average of Total Sunspot Area on the Sun's Visible Hemisphere {dateString}.", \
                   labelkwargs, y=1.05, color=colorFG)
    ax10.set_ylabel(r'Sunspot Area (m$^2$)', labelkwargs, labelpad=10, \
                    color=colorFG)
    ax10.set_ylim(scYLim)
    
     
    # Plotting UP ratio.
    ax20.errorbar(t, upRatioTot, yerr=upRatioTotE, fmt="o", ms=scSize, \
                  label="Average U/P Area Ratio", color=colorFG)
    #ax20.plot(t, movingAvg(upRatioTot, maWidth), lw=lw, color=colorFG)
    ax20.set_title(f"{maString} Moving Average of U/P Ratio on the Sun's Visible Hemisphere {dateString}.", \
                   labelkwargs, y=1.05, color=colorFG)
    ax20.set_ylabel('U/P Area Ratio', labelkwargs, labelpad=10, color=colorFG)
    ax20.set_ylim(upYLim)


    # Plotting sunspot number graph.
    #ax30.plot(t, movingAvg(uNum, maWidth), lw=lw, color=colorU, label="Umbra Number")
    #ax30.plot(t, movingAvg(pNum, maWidth), lw=lw, color=colorP, label="Penumbra Number")
    #ax30.scatter(t, sNum, marker=",", s=bfSize/5, lw=lw, color=colorFG, label="Sunspot Number")
    ssnMA = movingAvg(sNum, maWidth)
    
    print()
    maxSSN = max(ssnMA)
    maxSSNindex = list(ssnMA).index(maxSSN)
    maxSSNdate = t[maxSSNindex]
    print("Maximum daily SSN: ", maxSSN, ", Index: ", maxSSNindex, ", Date: ", maxSSNdate, "Avg Over: ", maString)


    minSSN = min(ssnMA)
    minSSNindex = list(ssnMA).index(minSSN)
    minSSNdate = t[minSSNindex]
    print("Minimum daily SSN: ", minSSN, ", Index: ", minSSNindex, ", Date: ", minSSNdate, "Avg Over: ", maString)
    print()
    
    ax30.plot(t, ssnMA, lw=lw, color=colorFG, label="Sunspot Number")
    ax30.set_title(f"{maString} Moving Average of Sunspot Number on the Sun's Visible Hemisphere {dateString}.", \
                   labelkwargs, y=1.05, color=colorFG)
    ax30.set_ylabel(r'Number of Occurrences', labelkwargs, labelpad=10, \
                    color=colorFG)
    ax30.set_ylim(bottom=0)
   
    # Format axes.
    axes = [ax00, ax10, ax20, ax30]
    formatAxes(axes)
    for ax in axes:
        ax.set_xlim(min(t), max(t)) 
        ax.set_xlabel('Date', labelkwargs, labelpad=10, color=colorFG)
        ax.tick_params(axis='x', labelrotation=45)
        ax.xaxis_date()
    
    #fig.legend()
    fig.tight_layout(pad=3) #, rect=[0, 0, 0.88, 1]
    
    # Save plot.
    if output["exportPlots"]:
        filename = savePlot("Butterfly.png", foldername)
        plt.savefig(filename, dpi=500, facecolor=colorBG)


def plotDistribution(allImgData, foldername, distType="area"):
    
    strTime = "Solar Cycle 24"
    
    # Unpacking relevant stats.
    if distType == "area":
        uAreas, pAreas = [], []
        for imgData in allImgData:
            for uArea in imgData["uAreas"][0]:
                uAreas.append(uArea)
            for pArea in imgData["pAreas"][0]:
                pAreas.append(pArea)
        histData = [uAreas, pAreas]
        histRange = (1, 200) #max(uAreas+pAreas)
        histStr = "Areas"
        histBins = 50
        print("Avg P Area: ", sum(pAreas)/len(pAreas))
        print("Avg U Area: ", sum(uAreas)/len(uAreas))
    
    elif distType == "lat":
        uLats, pLats = [], []
        for imgData in allImgData:
            for uLat in imgData["uLats"]:
                uLats.append(uLat)
            for pLat in imgData["pLats"]:
                pLats.append(pLat)
        uLats, pLats = [uLat*90 for uLat in uLats], [pLat*90 for pLat in pLats]
        histData = [uLats, pLats]
        histRange = (-60, 60)
        histStr = "Latitudes"
        histBins = 60
        
        pLatsAbs = [abs(pLat) for pLat in pLats]
        uLatsAbs = [abs(uLat) for uLat in uLats]
        print("Avg abs P Latitude: ", sum(pLatsAbs)/len(pLatsAbs))
        print("Avg abs U Latitude: ", sum(uLatsAbs)/len(uLatsAbs))
    
    # Plotting histogram of umbral & penumbral areas/lats.
    fig = plt.figure(figsize = (9, 6), facecolor = colorBG)
    ax = fig.add_subplot()
    
    ax.hist(histData, bins=histBins, range=histRange, histtype='bar', stacked=True,\
            label=[f"Umbrae", f"Penumbrae"], color=[colorU, colorP])
    
    # If area, show minimum lines.
    if distType == "area":
        ax.axvline(x=ip.pMinSize, label="Minimum Penumbral Area", color=colorP, \
                    lw=2, linestyle="dashed")
        ax.axvline(x=ip.uMinSize, label="Minimum Umbral Area", color=colorU, \
                    lw=2, linestyle="dashed")
  
    
    # Formatting
    ax.set_title(f"Histogram of Sunspot {histStr} During {strTime}.", \
                 labelkwargs, y=1.05, color=colorFG)
    if distType == "area":
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel('Area in Square Pixels', labelkwargs, labelpad=10, \
                      color=colorFG)
    elif distType =="lat":
        ax.set_xlabel(r'Latitude ($^\circ$)', labelkwargs, labelpad=10, \
                      color=colorFG)

    ax.set_ylabel('Number of Occurrences', labelkwargs, labelpad=10, color=colorFG)
    ax.set_xlim(histRange)
    
    ax.legend(loc="upper right")
    formatAxes([ax])
    fig.tight_layout(pad=3)

    # Save plot.
    if output["exportPlots"]:
        filename = savePlot(f"Distribution of {histStr}.png", foldername)
        plt.savefig(filename, dpi=500, facecolor=colorBG)


def printDataTable(allImgData, foldername):
    
    # Unpack the table data from the dictionary into list of lists.
    cols = []
    keys = ["uSum", "pSum", "sSum", "uPerPAvg", "uPerSAvg", "uAreasAll", \
            "pAreasAll", "upRatioTot"]
    for key in keys:
        col = []
        for imgData in allImgData:
            col.append(imgData[key])
        cols.append(col)
    
    radii, t = [], []
    for imgData in allImgData:
        radii.append(imgData["sunRadius"])
        ti = fileToDatetime(imgData["filepath"], foldername)
        t.append(ti)
    
    # Function for rounding number to number of sig figs.
    def roundSF(number:float, sf=output["sf"])->float:
        return f"{number:.{sf}g}"
    
    def formatData(data, sf=3, withError=True):
        if withError:
            num = data[0]
            err = data[1]
            err = roundSF(err, sf=sf)
        else:
            num = data
        num = roundSF(num, sf=sf)
        
        if withError:
            return num+" +/- "+err
        return num
    
    # Calculate average for list of results with error.
    def getAvg(col):
        avg = [np.nanmean(h) for h in np.array(col).T]
        avg = [np.nanmean(np.array(col).T[0])]
        Es = np.array(col).T[1]
        EsNoNans = Es[~np.isnan(Es)]
        EsAvg = np.sqrt(np.sum(np.square(EsNoNans)))/np.size(EsNoNans)
        avg.append(EsAvg)
        return avg
    
    # Convert data into metres, calculate averages, round to sf & format string.
    avgs = ["Mean"]
    for i, key in enumerate(keys):
        if key in ["uSum", "pSum", "sSum", "uPerPAvg", "uPerSAvg"]:
            avg = np.nanmean(np.array(cols[i]))
            avgs.append(formatData(avg, withError=False))
            cols[i] = [formatData(ele, withError=False) for ele in cols[i]]
            
        elif key in ["uAreasAll", "pAreasAll"]:
            cols[i] = [[pixToM(j, radii[k]) for j in ele] for k, \
                       ele in enumerate(cols[i])]
            avg = getAvg(cols[i])
            avgs.append(formatData(avg))
            cols[i] = [formatData(ele) for ele in cols[i]]
            
        elif key in ["upRatioTot"]:
            avg = getAvg(cols[i])
            avgs.append(formatData(avg))
            cols[i] = [formatData(ele) for ele in cols[i]]
                
    # Create PrettyTable
    x = prt()
    prt.title = f"Sunspot Data From {t[0]} to {t[-1]}"
    x.add_column("Image Datetime", t)
    for col, key in zip(cols, keys):
        x.add_column(key, col)
    x.add_row(avgs)
    
    if output["printDataTable"]:
        print(x)

    # Convert prt to csv.
    def exportPRT(table, filename, headers=True):
        raw = table.get_string()
        data = [tuple(filter(None, map(str.strip, splitline)))
                for line in raw.splitlines()
                for splitline in [line.split('|')] if len(splitline) > 1]
        if table.title is not None:
            data = data[1:]
        if not headers:
            data = data[1:]
        with open(filename, 'w') as f:
            for d in data:
                f.write('{}\n'.format(','.join(d)))
    
    # Export csv.
    if output["exportCSV"]:
        filename = savePlot("DataTable.csv", foldername)
        exportPRT(x, filename)



def main(allImgData, output, foldername):

    if output["plotVisual"]:
        plotVisual(allImgData, output, foldername)
                
    if output["plotAreaDistribution"]:
        plotDistribution(allImgData, foldername, distType="area")

    if output["plotLatDistribution"]:
        plotDistribution(allImgData, foldername, distType="lat")

    if output["plotButterfly"]:
        plotButterfly(allImgData, foldername)

    if output["printDataTable"] or output["exportCSV"]:
        printDataTable(allImgData, foldername)



