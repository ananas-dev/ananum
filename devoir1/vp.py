import math

d0 = 8 
d1 = 9
e = 2

vp_1 = (d0 +d1 + math.sqrt((d0 +d1)*(d0 +d1) - 4*(d1 * d0-e*e)))/2.0
vp_2 = (d0 +d1 - math.sqrt((d0 +d1)*(d0 +d1) - 4*(d1 * d0-e*e)))/2.0

print("voici les deux vp :" + str(vp_1) + " et " + str(vp_2))