#!/usr/bin/python
import string
import numpy as np
import json
import time, datetime, calendar
from pylab import *
from math import *
from scipy.cluster.vq import *
from scipy import stats
import gmplot

def onemode(values):
    return max(set(values), key=values.count)

def dt(u):
    return datetime.datetime.utcfromtimestamp(u)

def ut(d):
    return time.mktime(d.timetuple())

def isNaN(num):
    return num != num

fh = open("test_data_v3.1");

x = []
for line in fh.readlines():
    y = line.split()
    x.append(y)
fh.close()
usr = 3   # user index
xarr = np.array(x)
userid = np.unique(xarr[:,0])
userdata = xarr[xarr[:,0] == userid[usr]]   # select user data

ts = map(int, userdata[:,1])
unixts = [x / 1000 for x in ts]

location = [json.loads(jsonstr) for jsonstr in userdata[:,2]]

print(dt(unixts[0]) + datetime.timedelta(hours = location[0]['lng']/180*12))
print(dt(unixts[-1]) + datetime.timedelta(hours = location[-1]['lng']/180*12))

times = []
dates = []
lng = []
lat = []
hr = []
for i in range(len(unixts)):
    dtlocal = dt(unixts[i]) + datetime.timedelta(hours=location[i]['lng']/180*12)
    t = dtlocal.time()
    d = dtlocal.date()
    times.append(t)
    dates.append(d)
    lng.append(location[i]['lng'])
    lat.append(location[i]['lat'])
    hr.append(t.hour)
    # print(d, t, location[i]['lng'], location[i]['lat'])

udates = list(set(dates))

# grid and average data by hour
shape = (len(udates),24)
hvar = np.zeros(shape)
hlng = np.zeros(shape)
hlat = np.zeros(shape)
for di in range(len(udates)):
    ind = [i for i, x in enumerate(dates) if x == udates[di]]
    for i in ind:
        scatter(lng[i], lat[i], c=hr[i], alpha=0.5)
        clim(0,24)
    for h in range(24):
        dlng = [lng[i] for i in ind if hr[i] == h]
        dlat = [lat[i] for i in ind if hr[i] == h]
        v = np.var(dlng) + np.var(dlat)
        hvar[di,h] = v
        mdlng = sum(dlng) / float(len(dlng))
        mdlat = sum(dlat) / float(len(dlat))
        hlng[di,h] = mdlng
        hlat[di,h] = mdlat

# create a list of places and list of hour in local time
hlocs = []
l = []
lh =[]
ncent = 6

for di in range(len(udates)):
    print(udates[di])
    print(alendar.day_name[udates[di].weekday()])
    for h in range(24):
        if ~ (isNaN(hlng[di,h]) or isNaN(hlat[di,h]) ) and hvar[di,h] < 1e-7 and hvar[di,h] != 0:
            l.append((hlng[di,h], hlat[di,h]))
            lh.append(h)
            scatter(hlng[di,h], hlat[di,h], s=100, c=h, alpha=0.5)
            clim(0, 24)
            print((hlng[di,h],hlat[di,h]), '\t', h , '\t', hvar[di,h])
    print

colorbar()

print

# perform clustering on the places
centroids,_ = kmeans(l, ncent)
idx, _ = vq(l, centroids)

# find the cluster where user spent most time
cent1 = onemode(idx.tolist())
print(cent1)

# cluster where user spent second most time
cent2 = onemode([x for x in idx.tolist() if x != cent1])
print(cent2)

# plot these two clusters
figure()
for i in range(len(centroids)):
    print(centroids[i], sum(idx==i))
    if i == cent1 or i == cent2:
        lcent = [x for j, x in enumerate(l) if idx[j] == i]
        lhcent = [x for j, x in enumerate(lh) if idx[j] == i]
        t = centroids[i]
        scatter(t[0], t[1], s=500, alpha=0.5)
        for j in range(len(lcent)):
            scatter(lcent[j][0], lcent[j][1], s=100, c=lhcent[j], alpha=0.5)
            clim(0, 24)

home = centroids[cent1]

figure()
# eliminate home from the list of places, keep the day time places
l = [x for i, x in enumerate(l) if idx[i] != cent1 and lh[i] > 7 and lh [i] < 19]
lh = [x for i, x in enumerate(lh) if idx[i] != cent1 and lh[i] > 7 and lh [i] < 19]

# redo clustering, the densest cluster is work
centroids, _ = kmeans(l, ncent)
idx, _ = vq(l, centroids)
cent1 = onemode(idx.tolist())
print(cent1)
cent2 = onemode([x for x in idx.tolist() if x != cent1])
print(cent2)

scatter(home[0], home[1], s=1000, alpha=0.5)
for i in range(len(centroids)):
    print(centroids[i], sum(idx==i))
    if i == cent1:
        lcent = [x for j, x in enumerate(l) if idx[j] == i]
        lhcent = [x for j, x in enumerate(lh) if idx[j] == i]
        t = centroids[i]
        scatter(t[0], t[1], s=500, alpha=0.5)
        for j in range(len(lcent)):
            scatter(lcent[j][0], lcent[j][1], s=100, c=lhcent[j], alpha=0.5)
            clim(0, 24)

work = centroids[cent1]

print
print('userid', userid[usr], 'home', home, 'work', work)

colorbar()
show()

gmap = gmplot.GoogleMapPlotter(np.mean(lat), np.mean(lng), 4)
gmap.heatmap(lat, lng)
# gmap.scatter([home[1],work[1]], [home[0],work[0]], '#3B0B39', size=40, marker=False)
gmap.scatter([home[1], work[1]], [home[0], work[0]], 'b', marker=True)
gmap.draw("heatmap.html")
