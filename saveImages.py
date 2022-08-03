import os
import os.path as osp
import datetime
from sunpy.net.helioviewer import HelioviewerClient


def saveImages(startDate, endDate, imgPerDay, foldername):

    # Form list of dates between start and end date.
    dates = []
    for i in range(startDate.toordinal(), endDate.toordinal()):
        dates.append(str(datetime.date.fromordinal(i)).replace('-', '/'))
    
    hv = HelioviewerClient()    # Initialise HelioviewerClient
    
    # Loop through each date, retrieve imgPerDay images from that day, at
    # even spaced intervals.
    for j in range(len(dates)):
        print("day", j+1, "/", len(dates))
        for i in range(imgPerDay):
            print("image", i+1, "/", imgPerDay)
            date = (dates[j]+" "+str(int(i*24/imgPerDay)).zfill(2)+":00:00")
            file = hv.download_png(date, 1, "[SDO,HMI,continuum,512,512]")
            filename = osp.split(file)[-1]
            
            # Determining file save location.
            path = osp.join(os.getcwd(), foldername)
            if not osp.exists(path):
                os.makedirs(path)
            newfilename = osp.join(path, filename)
            
            # Error handling.
            try:
                os.rename(file, newfilename)
            except FileExistsError as e:
                print("file",newfilename,"already exists")
            except Exception as e:
                raise e
                
                
startDate = datetime.datetime.strptime("01/04/2022", "%d/%m/%Y")
endDate = datetime.datetime.strptime("01/05/2022", "%d/%m/%Y")
imgPerDay = 12

foldername = str(str(startDate)+" to "+str(endDate)+" ("+str(imgPerDay)+")")
foldername = foldername.replace(":", "-")

saveImages(startDate, endDate, imgPerDay, foldername)