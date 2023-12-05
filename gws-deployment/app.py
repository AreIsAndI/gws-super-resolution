import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import os
from classes import *

def genImg(pilImage):
    # load pretrained generator
    generator = GeneratorResNet()
    generator.eval()
    generator.load_state_dict(torch.load("Include/generator.pth",
                                        map_location=torch.device('cpu')))

    # do several configuration
    low_img = NewImage(pilImage.convert(mode='RGB'), (64, 64))
    gen_img = generator(low_img.tensorItem().unsqueeze(0))
    gen_img = make_grid(gen_img, nrow=1, normalize=True)
    to_pil = transforms.ToPILImage()
    generated = to_pil(gen_img)

    return generated

def main():
    # header
    st.header("GWS! - CCTV Super Resolution")

    # create custom layouts
    tab1, tab2 = st.tabs(['Take a screenshot', 'Choose an Image File'])

    # upsampling from a screen shot video
    with tab1:
        gambar = st.camera_input('Take a picture')
        if gambar:
            # st.image(gambar)
            image = Image.open(gambar)
            st.write('Crop The Images')
            cropped_img = st_cropper(image, realtime_update=True, box_color='#0000FF',
                                      aspect_ratio=(1, 1))

            col1, col2 = st.columns([2, 2])
            with col1:
                # preview the cropped image
                st.write("Preview")
                _ = cropped_img.thumbnail((64,64))
                st.image(cropped_img)

                # test if upsampling is pressed
                if st.button("Upsampling!", type='primary', key='ups1'):
                    with col2:
                        img = genImg(cropped_img)

                        # show the generated image
                        st.write('Generated Upsampling Image!')
                        st.image(img, width=256)
                        st.write('`Tip!` Save the generated image by pressing it with right mouse button, then **Save image as...**')

    # upsampling from a choosen image
    with tab2:
        # uploaded image
        gambar = st.file_uploader("Choose The CCTV Images", type=["jpg", "png"])

        # realtime crop image
        if gambar:
            image = Image.open(gambar)
            st.write('Crop The Images')
            cropped_img = st_cropper(image, realtime_update=True, box_color='#0000FF',
                                      aspect_ratio=(1, 1))

            col1, col2 = st.columns([2, 2])
            with col1:
                # preview the cropped image
                st.write("Preview")
                _ = cropped_img.thumbnail((64,64))
                st.image(cropped_img)

                # test if upsampling is pressed
                if st.button("Upsampling!", type='primary', key='ups2'):
                    with col2:
                        # run the generator function
                        img = genImg(cropped_img)

                        # show the generated image
                        st.write('Generated Upsampling Image!')
                        st.image(img, width=256)
                        st.write('`Tip!` Save the generated image by pressing it with right mouse button, then **Save image as...**')

if __name__ == '__main__':
    main()