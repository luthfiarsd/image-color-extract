import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Color Picker - Ekstrak Warna Dominan",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Mengkonversi nilai RGB (tuple) ke format hex
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# Mengekstrak warna dominan dari gambar menggunakan K-Means clustering
def extract_dominant_colors(image, k=5):
    # Konversi PIL Image ke numpy array
    img_array = np.array(image)

    # Jika gambar memiliki 4 channel (RGBA), konversi ke RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Reshape gambar menjadi array 2D (pixels x RGB)
    pixels = img_array.reshape(-1, 3)

    # Hapus pixel yang benar-benar hitam (0,0,0) untuk hasil yang lebih baik
    pixels = pixels[~np.all(pixels == 0, axis=1)]

    # Gunakan K-Means untuk mengelompokkan warna
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Dapatkan centroid (warna dominan)
    colors = kmeans.cluster_centers_

    # Hitung jumlah pixel untuk setiap cluster untuk mengurutkan berdasarkan dominansi
    labels = kmeans.labels_
    label_counts = np.bincount(labels)

    # Urutkan warna berdasarkan jumlah pixel (dari yang paling dominan)
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_colors = colors[sorted_indices]

    return sorted_colors, label_counts[sorted_indices]


def main():
    st.title("🎨 Color Picker - Ekstrak Warna Dominan")
    st.markdown("---")

    # Sidebar untuk pengaturan
    st.sidebar.header("⚙️ Pengaturan")
    num_colors = st.sidebar.slider(
        "Jumlah warna yang diekstrak:",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Pilih berapa banyak warna dominan yang ingin ditampilkan",
    )

    # Area upload file
    st.header("📁 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=["png", "jpg", "jpeg"],
        help="Upload gambar dalam format PNG, JPG, atau JPEG",
    )

    if uploaded_file is not None:
        try:
            # Load dan tampilkan gambar
            image = Image.open(uploaded_file)

            # Tampilkan informasi gambar
            st.header("🖼️ Gambar yang Diupload")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(
                    image,
                    caption=f"Gambar: {uploaded_file.name}",
                    use_container_width=True,
                )

            with col2:
                st.info(
                    f"""
                **Informasi Gambar:**
                - **Nama:** {uploaded_file.name}
                - **Ukuran:** {image.size[0]} x {image.size[1]} px
                - **Mode:** {image.mode}
                - **Format:** {image.format}
                """
                )

            # Tombol untuk memproses gambar
            if st.button("🚀 Ekstrak Warna Dominan", type="primary"):
                with st.spinner("Sedang menganalisis gambar..."):
                    # Ekstrak warna dominan
                    colors, counts = extract_dominant_colors(image, k=num_colors)
                    total_pixels = sum(counts)

                    cols = st.columns(num_colors)

                    for i, (col, color, count) in enumerate(zip(cols, colors, counts)):
                        rgb_values = tuple(map(int, color))
                        hex_code = rgb_to_hex(color)
                        percentage = (count / total_pixels) * 100

                        with col:
                            color_box = f"""
                            <div style="
                                background-color: {hex_code};
                                height: 100px;
                                border-radius: 10px;
                                border: 2px solid #ddd;
                                margin-bottom: 10px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            "></div>
                            """
                            st.markdown(color_box, unsafe_allow_html=True)

                            st.markdown(
                                f"""
                            **Warna #{i+1}**
                            - **Hex:** `{hex_code}`
                            - **RGB:** `{rgb_values}`
                            - **Dominansi:** {percentage:.1f}%
                            """
                            )

                            # Tabel detail warna
                    st.subheader("📋 Detail Warna")

                    # Buat data untuk tabel
                    color_data = []
                    for i, (color, count) in enumerate(zip(colors, counts)):
                        rgb_values = tuple(map(int, color))
                        hex_code = rgb_to_hex(color)
                        percentage = (count / total_pixels) * 100

                        color_data.append(
                            {
                                "Urutan": i + 1,
                                "Hex Code": hex_code,
                                "RGB": f"rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})",
                                "R": rgb_values[0],
                                "G": rgb_values[1],
                                "B": rgb_values[2],
                                "Jumlah Pixel": int(count),
                                "Persentase": f"{percentage:.2f}%",
                            }
                        )

                    # Tampilkan tabel
                    df = pd.DataFrame(color_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
            st.info("💡 Pastikan file yang diupload adalah gambar yang valid.")

    else:
        # Tampilkan contoh jika belum ada gambar yang diupload
        st.info("👆 Silakan upload gambar untuk memulai ekstraksi warna dominan.")


# Footer aplikasi
def add_footer():
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Luthfi Hamam Arsyada - 140810230007</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    add_footer()
