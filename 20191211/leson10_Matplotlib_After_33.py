import numpy as np
import matplotlib.pyplot as plt

# В строке интерпретатора 
# %matplotlib inline - рисовать во встроеном окне
# %matplotlib qt5 - рисовать в новом окне


#1 подход
fig = plt.figure()  # Полотно, канва
x = np.linspace(0, 10, 100) # Создается последовательность чисел

plt.subplot(2, 1, 1) # Количество рисунков (axes) (строки, колонки, номер рисунка по порядку)
plt.plot(x, np.sin(x))


plt.subplot(2, 1, 2) # Количество рисунков (axes) (строки, колонки, номер рисунка по порядку)
plt.plot(x, np.cos(x))
plt.show() # Показ рисунка


#2 подход
fig, ax = plt.subplots(2)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))


#3 несколько графиков на одном рисунке
plt.subplots(0)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

#4 маркеры и цвет
x = np.linspace(0, 10, 30)
y = np.sin(x)
#plt.plot(x, y, 'o', color='black')
#plt.plot(x, y, 'x', color='red')
plt.plot(x, y, 'x', color='red')
plt.show()

#5 маркеры
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(np.random.rand(5), np.random.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=5)
plt.xlim(0, 1.8) # Границы по x
plt.ylim(0,0.8) # Границы по y
plt.show()

#6 стиль линии
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, '.--k') # Стиль линии '.--k'  . - рисунок маленькой точкой -- - пунктир к - черный
plt.show()

#7
plt.plot(x, y, '-p', color='gray',
         markersize=10, linewidth=2,
         markerfacecolor='red',
         markeredgecolor='blue',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)

#8
x=np.linspace(-1,1,41) 
plt.plot(x,x,'r-s',x,x**2,'b--p',x,x**3,'kd', 
         linewidth=4, linestyle='')
plt.title('Титул графика' ,fontsize=20)

#9 разный масштаб по осям 
# отношение между единицами по осям set_aspect
x=np.linspace(0,1,101)
plt.plot(x,2*x)
plt.axes().set_aspect(0.5)

#10 масштабирование
x=np.linspace(0,40*np.pi,1000)
plt.plot(x,np.sin(x))
plt.axes().set_aspect(10)
#11 Подпись к оси х
plt.xlabel('Ось абсцисс',{'fontname':'Times New Roman',
                          'fontsize':'50'})
plt.ylabel('Ось ординат')

# Координатная сетка
plt.grid(True)

#12    
x = np.linspace(0, 5, 10)
y = x**2
fig = plt.figure(facecolor='blue' )
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8],facecolor='red') 
ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.3],facecolor='cyan') 
# Главный график
ax1.plot(x, y, '-or', linewidth=3)
ax1.set_title('title' )
# внутренний график
ax2.plot(y, x, '--.g', linewidth=2)
ax2.set_xlabel('y' )
ax2.set_ylabel('x' )
ax2.set_title('inner title' )
    
#13 Устанавливает равные соотношения высоты и ширины
plt.plot([0,1,2,0],[0,2,1,0],linewidth=4)
plt.axis('equal')

#14 круг
t=np.linspace(0,2*np.pi,100) 
plt.plot(np.cos(t),np.sin(t),linewidth=4) 
plt.axes().set_aspect(1)



#15 управление делениями
import matplotlib.ticker as ticker
x = np.linspace(-10, 10, 200)
y = 0.01*(x + 9)*(x + 6)*(x - 6)*(x - 9)*x
fig, ax = plt.subplots()
ax.plot(x, y, color = 'r', linewidth = 3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

# размер изображения
fig.set_figwidth(12)
fig.set_figheight(8)

#16 сравнительный анализ нескольких функций
x=np.linspace( -2*np.pi,2*np.pi,100)
y1=1/(1+x**2)
y2=np.cos(x)**2
y3=np.exp( -x**2/10)
z=2*(y1+y2+y3)
plt.stackplot(x,y1,y2,y3)
plt.plot(x,z,'k', linewidth=4)
plt.xlim( -2*np.pi , 2*np.pi)
plt.grid(True)


#17 Точечный график
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, marker='o')
plt.axis('equal')

#18 Продвинутый точечный график
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)

#19
x = np.random.rand(5000)
y1 = np.random.gamma(1, size = 5000)
y2 = np.random.gamma(2, size = 5000)
y3 = np.random.gamma(4, size = 5000)
y4 = np.random.gamma(8, size = 5000)
fig,ax= plt.subplots()
plt.scatter(x, y1,
           c = 'r',
           s = 1)
plt.scatter(x + 1, y2,
           c = [[0.1, 0.63, 0.55]],
           s = 1)
plt.scatter(x + 2, y3,
           c = '#ad09a3',
           s = 1)
plt.scatter(x + 3, y4,
           c = ['0.9'],
           s = 1)
ax.set_facecolor('black')

#20 Заполнение области под кривой (между нулем и функцией)
x = np.linspace(0, 1) 
y = np.sin(4 *np.pi*x)*np.exp(-5*x) 
plt.fill(x, y, 'r') 
plt.grid(True)



#21 График ошибок
xt=np.linspace(-4,4,101)
yt=1/(xt**2+1)
xe=np.linspace(-3,3,21)
yerr=0.1*np.ones(21)
ye=1/(xe**2+1)+yerr*np.random.normal(size=21)
plt.plot(xt,yt)
plt.errorbar(xe,ye,fmt='ro',yerr=yerr,
             ecolor='gray', 
             elinewidth=3, 
             capsize=2
             )

#22 График ошибок с усами по обеим координатам
x = np.linspace(0, 1, 15)
y = np.random.random_sample(15)
xerr = np.random.random_sample(15) / 10
yerr = np.random.random_sample(15) / 10
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o-', ecolor='red')



#23 Столбчиковая диаграма
x = np.arange(1, 8)
y = np.random.randint(1, 20, size = 7)
plt.bar(x, y, width=0.5,color='red')
plt.barh([6, 7, 8], [10, 15, 21],color='green')

#24
plt.barh(x,y)


#25 Цвета из массива
color_rectangle = np.random.rand(7, 3)
plt.bar(x, y, width=0.5,color=color_rectangle)


#26 Несколько объектов с разными значениями
data = np.random.randn(1000)
# plt.hist(data)
fig = plt.figure(facecolor='white') 
n=8 
data1=10*np.random.rand(n) 
data2=10*np.random.rand(n) 
data3=10*np.random.rand(n) 
locs = np.arange(1,n+1) 
wid = 0.3 # ширина столбца (приблизительно 1/3)
plt.bar(locs, data1, width=wid,color='blue') 
plt.bar(locs+wid, data2, width=wid, color='red') 
plt.bar(locs+2*wid, data3, width=wid, color='green') 
plt.grid(True)


#27 bottom нижний край прямоугольников
x = np.arange(1, 8)
y1 = np.random.randint(1, 10, size = 7)
y2 = np.random.randint(1, 10, size = 7)
y3 = np.random.randint(1, 10, size = 7)
plt.bar(x, y1)
plt.bar(x, y2, bottom = 11)
plt.bar(x, y3, bottom = 21)

#28 построение столбиков из суммы
x = np.arange(1, 8)
data_1 = np.random.randint(2, 15, size = 7)
data_2 = np.random.randint(3, 20, size = 7)
plt.bar(x, data_1)
plt.bar(x, data_2, bottom = data_1)

#29 пирог, не рекомендуют использовать, т.к. типа хуже воспринимается 
fig = plt.figure(facecolor='white') 
data=[8,2,5,3,6,4] 
lbls = ['apple', 'pear', 'orange', 'lemon', 'cherries', 'currants']
plt.pie(data,labels = lbls,textprops={'fontsize': 10},
        autopct='%.2f')
plt.axis('image')

data=np.random.randn(10000)
#30 гистограмма 
# density относительные или абсолютные значения 
# True - относительные
# False - абсолютные
plt.hist(data)
plt.hist(data, bins=50, density=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='green')

plt.hist(data, bins=50, density=False, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='green')

#31 Сравнение гистограмм
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

#32 
data2=np.random.normal(0, 0.8, 1000)
grid = plt.GridSpec(1, 4,wspace=0.5)
sc_plot=plt.subplot(grid[0, 0:3])
#sc_plot.scatter(range(1000),data2)
sc_plot.plot(range(1000),data2)
sc_plot.set_xlabel("Номер спостереження")
sc_plot.set_ylabel('Значення спостереження')
hg_plot=plt.subplot(grid[0, 3])
hg_plot.hist(data2,bins=75,orientation='horizontal')

#33

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax=Axes3D(fig) 
t=np.linspace(-4*np.pi,4*np.pi,100) 
r=t**2/80+1 
x=r*np.cos(t) 
y=r*np.sin(t) 
z=t/(2*np.pi) 
ax.plot(x,y,z,linewidth=4)

#34
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure() 
ax=Axes3D(fig) 
n = 60
xs = np.random.rand(n) 
ys = np.random.rand(n) 
zs = np.random.rand(n) 
ax.scatter(xs, ys, zs, c='r', marker='o',s=200)


#35
fig=plt.figure() 
ax=Axes3D(fig) 
u=np.linspace(-4*np.pi,4*np.pi,100) 
x,y=np.meshgrid(u,u) 
r=np.sqrt(x**2+y**2) 
z=np.sin(r)/r 
ax.plot_wireframe(x, y, z, rstride=1, cstride=1)

#35a
import matplotlib as mpl
surf=ax.plot_surface(x,y,z,rstride=1,cstride=1,
                     linewidth=0, cmap=mpl.cm.hsv) 
fig.colorbar(surf, shrink=0.75, aspect=15)




#36
 
import matplotlib.animation as animation 
fig = plt.figure(facecolor='white') 
ax = plt.axes(xlim=(0, 8),ylim=(-1, 1) ) 
line, = ax.plot([ ], [ ], lw=3) 
def redraw(i): 
    x = np.linspace(0, 8, 200) 
    y = np.sin(i * x/10)/(1+x**2) 
    line.set_data(x, y)
anim =animation.FuncAnimation(fig,redraw,frames=500,interval=10)


#37
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
vix1 = pd.read_csv('Data/VIXCLS.csv', index_col=0,  na_values='.',
                  parse_dates=True,
                  squeeze=True).dropna()
ma = vix1.rolling('90d').mean()
ma.plot(
    color='black', linewidth=1.5, marker='', figsize=(8, 4), label='VIX 90d MA'
    )
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('90-дневное скользящее среднее')
#ax.set_title('Volatility Regime State')
ax.legend(loc='upper center')
ax.set_xlim(xmin=ma.index[0], xmax=ma.index[-1])
ax.axhline(
    vix1.mean(), linestyle='dashed', color='xkcd:dark grey',
    alpha=0.6, label='Full-period mean', marker=''
)
