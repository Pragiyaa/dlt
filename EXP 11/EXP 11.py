importnumpyasnp
importmatplotlib.pyplotasplt
import tensorflow as tf
fromtensorflow.keras.modelsimportSequential
fromtensorflow.keras.layersimportDense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU
fromtensorflow.keras.optimizersimportAd
am # Load and normalize MNIST data
(x_train,_),(_,_)=tf.keras.datasets.mnist.load_data()
x_train=(x_train.astype(np.float32)-127.5)/127.5#Normalizeto[-1,1]
x_train = np.expand_dims(x_train, axis=-1)
# Define Generator
defbuild_generator(
):
model= Sequential([
Dense(128*7*7,input_dim=100),
LeakyReLU(0.2),
Reshape((7,7,128)),
Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'
), LeakyReLU(0.2),
Conv2DTranspose(1,kernel_size=4,strides=2,padding='same',activation='tanh')
])
returnmodel
# Define Discriminator
defbuild_discriminator
():
model= Sequential([
Conv2D(64,kernel_size=4,strides=2,padding='same',input_shape=(28,28,1)),
LeakyReLU(0.2),
Flatten(),
Dense(1,activation='sigmoid')
])
returnmodel
# Build and compile models
generator = build_generator()
discriminator=build_discriminato
r()
discriminator.compile(optimizer=Adam(0.0002),loss='binary_crossentropy',metrics=['acc
uracy'])
# Combined GAN model
discriminator.trainable =
Falsegan_input=tf.keras.Input(shape
=(100,)) generated_img =
generator(gan_input)
gan_output=discriminator(generated_im
g)
gan=tf.keras.Model(gan_input,gan_output
)
gan.compile(optimizer=Adam(0.0002),loss='binary_crossent
ropy') # Training function
deftrain_gan(epochs=1000,batch_size=128):
half_batch = batch_size // 2
forepochinrange(epochs):
#Traindiscriminator
idx=np.random.randint(0,x_train.shape[0],half_batch)
real_imgs = x_train[idx]
noise=np.random.normal(0,1,(half_batch,100))
fake_imgs = generator.predict(noise, verbose=0)
d_loss_real = discriminator.train_on_batch(real_imgs,
np.ones((half_batch, 1)))
d_loss_fake=discriminator.train_on_batch(fake_imgs,np.zeros((half_batc
h,1))) # Train generator
noise=np.random.normal(0,1,(batch_size,100))
valid_y = np.ones((batch_size, 1))
g_loss=gan.train_on_batch(noise,valid_
y)
# Display progress every 200 epochs
ifepoch%200==0:
print(f"Epoch{epoch}—Dloss:{d_loss_real[0]:.4f},Gloss:{g_loss:.4f}")
show_images()
#Visualizegeneratedimages
defshow_images():
noise = np.random.normal(0, 1, (16, 100))
gen_imgs = generator.predict(noise,
verbose=0)
gen_imgs=0.5*gen_imgs+0.5#Rescaleto[0,1]
fig, axs = plt.subplots(4, 4, figsize=(4, 4))
cnt= 0
foriinrange(4):
forjinrange(4):
axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray
') axs[i, j].axis('off')
cnt += 1
plt.tight_layout()
plt.show()
#Runtraining
train_gan(epochs=1000,batch_size=128)