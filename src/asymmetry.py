# from re import I
# from turtle import color
# from cgi import test
import cv2#, platform,time
# from cv2 import waitKey
# from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.image as mpimg
# from PIL import Image 
# from scipy import ndimage

def asymetrie(filename,r,seuil):
    benin=False

    plt.figure(1)
    I=cv2.imread(filename,0)#.astype(float)
    plt.subplot(221)
    #prétraitement de l'image
    plt.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))
    plt.title('image originale')
    ret,thresh1=cv2.threshold(I,seuil,255,cv2.THRESH_BINARY)
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(thresh1,cv2.COLOR_BGR2RGB))
    plt.title('image seuillée')

    S=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    I_open=cv2.dilate(cv2.erode(thresh1,S),S)
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(I_open,cv2.COLOR_BGR2RGB))
    plt.title('image après ouverture')
    #afin d'avoir le bon type
    TYPE=type(I)

    IMAGE=I_open.astype(TYPE)
    IMAGE2=I_open
    #partie snake

    Lx, Ly = np.shape(IMAGE2)

    # %%
    ###Creation du snake###
    centre=[int(Ly/2),int(Lx/2)]
    rayon=min(int((Lx-5)/2), int((Ly-5)/2))/r

    K = 1000
    snakeX = []
    snakeY = []
    pas = (2*np.pi)/K
    for i in range(K):
        theta = i*pas
        snakeX = np.append(snakeX, int(centre[0] + rayon * np.cos(theta)))
        snakeY = np.append(snakeY, int(centre[1] + rayon * np.sin(theta)))
    # print(snakeX.shape)
    c = np.zeros((K,1,2))
    # print(c.shape)
    c[:,:,0] = snakeX.reshape((K,1))
    c[:,:,1] = snakeY.reshape((K,1))
    #c = np.concatenate((snakeX.reshape((K,1)),snakeY.reshape((K,1))),axis=2)
    # print(c[:,0,0])
    contour_list = []
    contour_list.append(c.astype(int))
    snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
    plt.imshow(snake)
    plt.show()



    # %%
    ### Parametres ###
    alpha = 3
    beta = 0.1
    gamma = 1.2

    # %% [markdown]
    # #### On défini les matrices pour les opérateurs
    # La matrice `D1` correspond à une approximation de la dérivée par différence finies. La matrice `D2` correspond à une dérivée seconde et `D4` à une dérivée quatrième.

    # %%
    ###Creation de D2, D4, D et A###
    Id = np.identity(K)
    D1 = np.roll(Id, 1, axis=-1) + Id*(0) - np.roll(Id,-1, axis=1)
    D2 = np.roll(Id, -1, axis=1) + Id*(-2) + np.roll(Id,1, axis=1)
    D4 = (np.roll(Id, -1, axis=1) + np.roll(Id,1, axis=1))*-4 + (np.roll(Id, -2, axis=1) + np.roll(Id,2, axis=1)) + Id*(6)
    D = alpha*D2 - beta*D4
    A = np.linalg.inv(Id - D)
    #logging.info('Operators generated')

    # %%
    # Le Gradient
    [Gx,Gy] = np.gradient(IMAGE2.astype(float))
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(Gx)
    plt.subplot(1,3,2)
    plt.imshow(Gy)
    Gx_norm = Gx/np.max(Gx)
    Gy_norm = Gy/np.max(Gy)
    NormeGrad = np.square(Gx_norm)+np.square(Gy_norm)

    plt.subplot(1,3,3)
    plt.imshow(NormeGrad,'gray')
    plt.show()
    #NormeGrad = NormeGrad*20
    # Gradient de la norme 
    [GGx,GGy] = np.gradient(NormeGrad.astype(float))

    #logging.info('Gradient computed')

    # %%
    # Algo ITERATIF
    limite = 3000
    iteration = 0
    nbfigure = 1

    Energie = list()
    energie_ela = list()
    energie_courb = list()
    enregie_ext = list()

    MEMORY = []
    Xn = snakeX
    Yn = snakeY
    MEMORY.append([Xn,Yn])
    #logging.info('Algorithm initialized')

    # %%
    flag = True
    while flag or (iteration < limite):
        # itération du SNAKE
        Xn1 = np.dot(A, Xn + gamma*GGx[Yn.astype(int),Xn.astype(int)] )
        Yn1 = np.dot(A, Yn + gamma*GGy[Yn.astype(int),Xn.astype(int)] )     
        Xn = Xn1
        Yn = Yn1   
        MEMORY.append([Xn,Yn])
        # Calcul de l'energie
        ELA = 0
        COURB  = 0
        EXT = 0
        Xnprime = np.dot(D1, Xn)
        Ynprime = np.dot(D1, Yn)
        Xnseconde = np.dot(D2, Xn)
        Ynseconde = np.dot(D2, Yn)
        for k in range(K):
            ELA += alpha*0.5*np.sqrt(np.square(Xnprime[k]) + np.square(Ynprime[k]))
            COURB += beta*0.5*np.sqrt(np.square(Xnseconde[k]) + np.square(Ynseconde[k]))
            EXT -= np.square(NormeGrad[int(Yn[k]),int(Xn[k])])
        Energie.append(ELA+COURB+EXT)
        enregie_ext.append(EXT)
        energie_courb.append(COURB)
        energie_ela.append(ELA)

        # Flag de sortie
        # 
        # TODO : 
        #   Calculer energie sur fenetre glissante pour flag de sortie afin de lisser les variations
        # if iteration > 200:
        #     nbSplit = iteration // 50
        #     EnerSplit = np.split(Energie, nbSplit)
        #     e1 = EnerSplit[-1]
        #     e2 = EnerSplit[-2]
        if (abs(Energie[iteration]-Energie[iteration-1])/Energie[iteration]<10):
            flag = False

        
        # Affichage
        if iteration % 10 == 0:
            c = np.zeros((K,1,2))
            c[:,:,0] = Xn1.reshape((K,1))
            c[:,:,1] = Yn1.reshape((K,1))
            contour_list = []
            contour_list.append(c.astype(int))
            snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
            # Sauvegarde des images pour faire l'animation
            #filename = f"img_{iteration:05d}.png"
            
            #cv2.imwrite(filename, snake)

        # Fin de la boucle
        iteration += 1


    # %%
    c = np.zeros((K,1,2))
    print(c.shape)
    c[:,:,0] = Xn1.reshape((K,1))
    c[:,:,1] = Yn1.reshape((K,1))
    contour_list = []
    contour_list.append(c.astype(int))
    snake = cv2.drawContours(image=cv2.cvtColor(IMAGE2, cv2.COLOR_GRAY2BGR),contours=contour_list, contourIdx=-1, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
    plt.imshow(snake)
    #cv2.imwrite("iteration_finale8.png",snake)
    plt.title('Itération finale')



    #inutile mais permet de mieux se repérer dans le snake
    for i in range(len(Xn)):
        if (i<250):
            color= "red"
        elif (i>=250 and i < 500):
            color="green"
        elif (i>=500 and i<750):
            color="blue"
        elif (i>=750 and i<1000):
            color="yellow"
        plt.plot(Xn[i],-Yn[i],marker="o",color=color)
        
    plt.title('snake en couleur')
    plt.show()
    #print(Xn)
    #print(Yn)

    #partie affichant le snake en couleur pas si important
    A=[Xn[140],-Yn[140]]
    B=[Xn[200],-Yn[200]]
    C=[Xn[700],-Yn[700]]
    D=[Xn[900],-Yn[900]]
    plt.plot(Xn[140],-Yn[140],marker="o",color="red")
    plt.plot(Xn[200],-Yn[200],marker="o",color="green")
    plt.plot(Xn[700],-Yn[700],marker="o",color="blue")
    plt.plot(Xn[900],-Yn[900],marker="o",color="yellow")
    plt.plot([A[0],C[0]],[A[1],C[1]],color="red")
    plt.plot([B[0],D[0]],[B[1],D[1]],color="blue")

    # plt.show()

    print("size Iopen : " +str(np.shape(I_open)))
    #seuillage de l'image pour la detection des cercles
    src_img=I>=seuil

    src_img=src_img.astype(np.uint8)
    src_img=src_img*255

    color_img=src_img
    print("size color_img : " + str(np.shape(color_img)))
    rows=src_img.shape[0]

    #utilisation de la fonction cv2.HoughCircles pour detecter les cercles
    circles_img = cv2.HoughCircles(src_img,cv2.HOUGH_GRADIENT,1,rows/8,param1=254,param2=10,minRadius=0,maxRadius=0)
    circles_img = np.uint16(np.around(circles_img))
    print(circles_img)
    #trace tous les cercles detectes et leurs centres
    for i in circles_img[0,:]:
        cv2.circle(color_img,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(color_img,(i[0],i[1]),2,(0,0,255),3)
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
    plt.title('Detection des cercles')
    plt.figure(3)
    plt.imshow(I)
    plt.plot(circles_img[0,0,0],circles_img[0,0,1],marker="o",color="red")
    plt.title('Tracé du cercle et du centre')
    #h etant un offset de 20 pixels pour d'autres critères
    h=20
    #affiche les paramètres pour la boite englobante
    #print(min(Xn),min(Yn))
    #print(max(Xn),max(Yn))
    #recadrer l'image par rappport à la boite englobante
    plt.figure(4)
    cropped=I_open[min(Yn-h).astype(int):max(Yn+h).astype(int),min(Xn-h).astype(int):max(Xn+h).astype(int)]
    plt.imshow(cropped,'gray')
    plt.title('Image recadrée')
    #test hauteur/largeur
    #calcul de l'ecart entre la hauteur et la largeur
    ecart=abs((max(Xn).astype(int)-min(Xn).astype(int))-(max(Yn).astype(int)-min(Yn).astype(int)))
    #si l'écart est impair, on l'augmente de 1
    if (ecart%2==1):
        ecart=ecart+1
    print(ecart)
    #toutes les conditions permettant les différents cas pour recadrer notre image pour qu'elle soit carrée, en fontion de l'écart
    if (max(Xn)-min(Xn)>max(Yn)-min(Yn)):
        cropped=I_open[min(Yn.astype(int)-h-(ecart/2).astype(int)).astype(int):max(Yn.astype(int)+h+(ecart/2).astype(int)).astype(int),min(Xn.astype(int)-h).astype(int):max(Xn.astype(int)+h).astype(int)]
        h,w=np.shape(cropped)
        print(h,w)
        #nouvelles séries de test pour savoir si elle est bien de taille carré
        if(h==w):
            print("c'est ok")
            #print("b",h*w)
        elif(h<w):
            h=w
            cropped=cv2.resize(cropped,(h,w))
        elif(h>w):
            w=h
            cropped=cv2.resize(cropped,(h,w))
        print("parfait : ",h,w)  
    elif (max(Xn)-min(Xn)<max(Yn)-min(Yn)):
        cropped=I_open[min(Yn.astype(int)-h).astype(int):max(Yn.astype(int)+h).astype(int),min(Xn.astype(int)-h-(ecart/2).astype(int)).astype(int):max(Xn.astype(int)+h+(ecart/2).astype(int)).astype(int)]
        h,w=np.shape(cropped)
        print(h,w)
        if(h==w):
            print("c'est ok")
            print("w",h*w)
        elif(h<w):
            h=w
            cropped=cv2.resize(cropped,(h,w))
        elif(h>w):
            w=h
            cropped=cv2.resize(cropped,(h,w))
        print("e",h,w) 
    elif (max(Xn)-min(Xn)==max(Yn)-min(Yn)):
        h,w=np.shape(cropped)
        print("c'est ok")
        print("d",h*w)

    plt.figure(5)
    plt.imshow(cropped,'gray')
    plt.title('Image recadrée de taille carrée')
    #on tourne notre image de 90 degres ou bien de 180 degrés on a le choix entre cv2.rotate_180 et cv2.rotate_90_clockwise ou cv2.rotate_90_counterclockwise
    cropped_rotate = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    #cropped_rotate = ndimage.rotate(cropped, 45)
    #on affiche notre image originale recadrée et celle tourner 
    plt.figure(6)
    plt.imshow(cropped_rotate,'gray')
    plt.title('Image recadrée carrée tournée de 90 degres')
    print(np.shape(cropped))
    print(np.shape(cropped_rotate))
    #on applique l'union moins l'intersection
    final=cv2.bitwise_or(cropped_rotate,cropped,mask=None)-cv2.bitwise_and(cropped_rotate,cropped,mask=None)


    #dice
    inter=cv2.bitwise_or(cropped_rotate,cropped,mask=None)
    union=cv2.bitwise_and(cropped_rotate,cropped,mask=None)
    plt.figure(7)
    plt.imshow(inter,'gray')
    plt.title("intersection entre l'image cropped et celle tournée de 90 degres")

    cpta=0
    for i in range(h):
        for j in range(w):
            if(cropped[i,j]==0):
                cpta+=1
    cptb=0
    for i in range(h):
        for j in range(w):
            if(inter[i,j]==0):
                cptb+=1
    cptc=0
    for i in range(h):
        for j in range(w):
            if(cropped_rotate[i,j]==0):
                cptc+=1


    print("ensemble A : ",cpta,"ensemble intersection : ",cptb,"ensemble B : ",cptc)
    dice=2*cptb/(cpta+cptc)

    print("dice : ",dice)
    if (dice>0.9):
        print("c'est benin pour dice")
        benin=True
    else:
        print("c'est malin pour dice")
        benin=False
    plt.figure(8)
    plt.imshow(final,'gray')
    plt.title("union - intersection")
    #on recupere les contours et on calcule leur aire
    cnt, hierarchy=cv2.findContours(final,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    cnt2=cnt[0]
    area= cv2.contourArea(cnt2)
    print("l'aire est de  : ",area)
    cpt=0
    rest=h*w
    print(final.size)
    #print(final)
    #on calcule les pixels de l'image qui sont dans le contour
    cpt1=0
    for i in range(h):
        for j in range(w):
            if(final[i,j]==0):
                cpt1+=1


    for i in range(h):
        for j in range(w):
            if(final[i,j]>=254):
                cpt+=1
                rest-=1
    print("le nombre de pixels blancs est de : ",cpt)
    print("le nombre de pixels restants est de : ",rest)

    #on calcule le ratio entre les pixels blancs et les pixels restants
    ratio=cpt/cpt1
    print(ratio)
    critere=0.93
    ratio_final=1-ratio
    if(ratio_final>critere):
        print("c'est benin pour ratio")
        benin=True
    elif(ratio_final<critere):
        print("c'est malin pour ratio")
        benin=False
    print("ratio_final : ",ratio_final)



    plt.figure(9)
    plt.imshow(union,'gray')
    plt.title("union")


    #partie vrais positifs ...
    #l'intersection entre les deux images contient les vrais positifs
    #calcul du nombre de pixels true positifs
    cpt_tp=0
    t1,s1=np.shape(inter)
    for i in range(t1):
        for j in range(s1):
            if(inter[i,j]==0):
                cpt_tp+=1

    print("true positifs :",cpt_tp)
    #calcul du nombre de pixels true negatifs
    cpt_tn=0
    t2,s2=np.shape(inter)
    for i in range(t2):
        for j in range(s2):
            if(inter[i,j]>=1):
                cpt_tn+=1

    print("true negatifs :",cpt_tn)

    I_test=255-union


    #false negatifs
    fn=I_test-(255-cropped_rotate)
    plt.figure(10)
    plt.imshow(fn,'gray')
    plt.title("false negatifs")


    cpt_fn=0
    t3,s3=np.shape(fn)
    for i in range(t3):
        for j in range(s3):
            if(fn[i,j]>=1):
                cpt_fn+=1

    print("false negatifs :",cpt_fn)





    #false positifs
    fp=I_test-(255-cropped)
    plt.figure(11)
    plt.imshow(fp,'gray')
    plt.title("false positifs")


    cpt_fp=0
    t3,s3=np.shape(fn)
    for i in range(t3):
        for j in range(s3):
            if(fp[i,j]>=1):
                cpt_fp+=1

    print("false positifs :",cpt_fp)

    #plt.figure()
    #plt.imshow(I_test,'gray')
    #plt.title("test")

    #calcul de précision
    precision=cpt_tp/(cpt_tp+cpt_fp)
    print("la precision est de : " ,precision)
    if (precision>0.9):
        print("c'est benin pour precision")
        benin=True
    else:
        print("c'est malin pour precision")
        benin=False
    print("finalement le gdb est : ",benin)
    plt.show()
    return benin




def main():
    filename = '/home/timothee/Documents/Mole-Checking/src/color3.jpg'
    if (filename=='data/color2.jpg'):
        seuil=125
        r=1.5
    elif(filename=='data/gdb_benin.jpg'):
        seuil=125
        r=3
    elif (filename=='/home/timothee/Documents/Mole-Checking/src/color3.jpg'):
        seuil=90
        r=1
    elif (filename=='data/color.jpg'):
        seuil=150
        r=2
    asymetrie(filename,r,seuil)
    pass

if __name__ == '__main__':
    main()