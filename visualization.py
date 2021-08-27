"""matplotlib을 이용한 visualization코드 작성
"""
import matplotlib.pyplot as plt
import numpy as np



from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


cost_array = np.array([
    [32832.99, 33114.03, 32671.03, 32743.96, 2197.274629],
    [32952.60, 33185.25, 32762.50, 32784.33, 2033.699532]
])
time_array = np.array([1,2,3,4,5])

plt.plot(time_array[:],cost_array[0,:],marker='o', markersize=5)
plt.plot(time_array[:],cost_array[1,:],marker='o', markersize=5)

plt.xlabel('시간',fontsize = 10)
plt.ylabel('가격',fontsize = 10)

plt.show()







    #
pass