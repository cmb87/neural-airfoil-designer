# ================
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
import json
from scipy.interpolate import splprep, splev

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import parse_args


def main(arg):

    for ctr,d in enumerate(glob.glob("./data/raw/*.dat")):
    
        df = pd.read_table(d, sep="\s+", skipfooter=1)


        if len(df.columns) == 1:
            df = df.reset_index(level=0)

        print(df)

        if True:
        #  df.columns = ["x", "y"] 

            try:
                df = df[df.iloc[:,0].notna()]
                df = df[df.iloc[:,1].notna()]
                df = df[df.iloc[:,0].apply(lambda x: isinstance(x, (int, np.int64, float)))]
                df = df[df.iloc[:,1].apply(lambda x: isinstance(x, (int, np.int64, float)))]
                df = df[(df.iloc[:,0]<=1.0) & (df.iloc[:,1]<=1.0) & (df.iloc[:,0]>=0.0) & (df.iloc[:,1]>=-1.0)]

            except:
                print(f"Couldnt process {d}")

            

            ntepts = 0
            eps = 1e-8

            print("=========",d,"========")

            while ntepts == 0:

                df["teidx"] = np.abs(df.iloc[:,0]-1.0)< eps
                ntepts = df.teidx.sum()
                eps *= 1.1


            nlepts = 0
            eps = 1e-8

            while nlepts == 0:

                df["leidx"] = np.abs(df.iloc[:,0]) < eps
                
                nlepts = df.leidx.sum()
                eps *= 1.1
            

            print(ntepts, nlepts)


            # Split into SS and PS
            x = df.iloc[:,0].to_numpy()
            y = df.iloc[:,1].to_numpy()

            teidx = df.index[df['teidx']].tolist()
            leidx = df.index[df['leidx']].tolist()


            if ntepts == 2 and nlepts ==1:

                x1, x2 = x[teidx[0]:leidx[0]+1][::-1], x[leidx[0]:]
                y1, y2 = y[teidx[0]:leidx[0]+1][::-1], y[leidx[0]:]


            # Case 2
            elif ntepts == 2 and nlepts ==2:

                # [1, 50] [49, 98]

                # [60, 61] [0, 119]
                

                # Case 2a
                if leidx[1] - teidx[0] == 1:

                    x1, x2 = x[leidx[0]-1:teidx[0]], x[leidx[1]-1:]
                    y1, y2 = y[leidx[0]-1:teidx[0]], y[leidx[1]-1:]
                    case = "2a"

                # Case 2b
                elif leidx[1] - leidx[0] == 1:

                    x1, x2 = x[teidx[0]:leidx[0]+1][::-1], x[leidx[1]:]
                    y1, y2 = y[teidx[0]:leidx[0]+1][::-1], y[leidx[1]:]
                    case = "2b"

                    


            # Case 3
            elif ntepts == 1 and nlepts ==2:

                print(leidx, teidx)
                # 3a: [57, 58] [1]
                # 3b: [1, 68] [67]
                # 3c: [1, 41] [81] , [1, 36] [67]
                # 
                if leidx[1]-leidx[0] == 1:

                    x1, x2 = x[teidx[0]:leidx[0]][::-1], x[leidx[1]:]
                    y1, y2 = y[teidx[0]:leidx[0]][::-1], y[leidx[1]:]
                    case = "3a"

                elif leidx[1]-teidx[0] == 1:

                    x1, x2 = x[leidx[0]:teidx[0]], x[leidx[1]:]
                    y1, y2 = y[leidx[0]:teidx[0]], y[leidx[1]:]
                    case = "3b"

                else:

                    teidx = [leidx[1]-1, teidx[0]]


                    # Case 2a
                    if leidx[1] - teidx[0] == 1:

                        x1, x2 = x[leidx[0]-1:teidx[0]-1], x[leidx[1]-1:]
                        y1, y2 = y[leidx[0]-1:teidx[0]-1], y[leidx[1]-1:]

                        case = "2a"

                    # Case 2b
                    elif leidx[1] - leidx[0] == 1:

                        x1, x2 = x[teidx[0]:leidx[0]+1][::-1], x[leidx[1]:]
                        y1, y2 = y[teidx[0]:leidx[0]+1][::-1], y[leidx[1]:]
                        case = "2b"



            legap = np.sqrt((x1[0] - x2[0])**2 + (y1[0] - y2[0])**2)
            tegap = np.sqrt((x1[-1] - x2[-1])**2 + (y1[-1] - y2[-1])**2)

            if legap < 1e-4:
                pass
            elif legap < 0.14:
                xmean = 0.5*(x1[0]+ x2[0])
                ymean = 0.5*(y1[0]+ y2[0])

                x1 = np.concatenate((np.asarray([xmean]), x1))
                x2 = np.concatenate((np.asarray([xmean]), x2))
                y1 = np.concatenate((np.asarray([ymean]), y1))
                y2 = np.concatenate((np.asarray([ymean]), y2))

            else:
                print("Warning:", d)
                continue

            if tegap < 1e-4:
                pass
            elif tegap < 0.14:
                xmean = 0.5*(x1[-1]+ x2[-1])
                ymean = 0.5*(y1[-1]+ y2[-1])

                x1 = np.concatenate((x1, np.asarray([xmean])))
                x2 = np.concatenate((x2, np.asarray([xmean])))
                y1 = np.concatenate((y1, np.asarray([ymean])))
                y2 = np.concatenate((y2, np.asarray([ymean])))

            else:
                print("Warning:", d)
                continue

            # Spline reconstruction
            try:
                tck1, _ = splprep([x1, y1], s=0,k=1)
                tck2, _ = splprep([x2, y2], s=0,k=1)

                u = np.linspace(0,1, arg.ih)

                xss = splev(u, tck1)
                xps = splev(u, tck2)


            except:
                print("Spline Error" , d)
                continue

            fp, fn = os.path.split(d)
            name = fn.split('.')[0]
            
            plt.plot(x1,y1,'b-')
            plt.plot(x2,y2,'r-')
            plt.plot(x1[0],y1[0],'bs')
            plt.plot(x2[0],y2[0],'rs')
            plt.plot(x1[-1],y1[-1],'bo')
            plt.plot(x2[-1],y2[-1],'ro')
            plt.plot(xss[0], xss[1], 'k--')
            plt.plot(xps[0], xps[1], 'k--')

            plt.axis("equal")
            plt.title(f"{d} {case}")
            plt.savefig(f"./data/images/{name}.png")


            plt.close()

            data = {"name": name, "ss": [xss[0].tolist(), xss[1].tolist()], "ps": [xps[0].tolist(), xps[1].tolist()]}

            with open(f"./data/processed/{name}.json", "w") as outfile:
                outfile.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    arg = parse_args()
    main(arg)