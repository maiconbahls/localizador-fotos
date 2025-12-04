import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import zipfile
import io

# Configuração da página
st.set_page_config(page_title="Localizador de Fotos", page_icon="📸", layout="wide")

st.title("📸 Localizador de Fotos com IA")
st.markdown("Faça upload de uma foto sua e das fotos do evento. A IA vai encontrar você!")

# --- FUNÇÕES AUXILIARES ---
def carregar_imagem_upload(uploaded_file):
    """Converte o arquivo de upload (bytes) para formato que a IA entende"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        return np.array(image)
    return None

# --- BARRA LATERAL (REFERÊNCIA) ---
st.sidebar.header("1. Quem vamos procurar?")
opcao_ref = st.sidebar.radio("Como enviar sua foto?", ["📁 Enviar Arquivo", "📷 Usar Webcam"])

imagem_referencia = None

if opcao_ref == "📁 Enviar Arquivo":
    ref_file = st.sidebar.file_uploader("Sua foto de rosto", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        imagem_referencia = carregar_imagem_upload(ref_file)
        st.sidebar.image(imagem_referencia, caption="Sua Foto (Referência)", width=200)

elif opcao_ref == "📷 Usar Webcam":
    camera_file = st.sidebar.camera_input("Tire uma foto agora")
    if camera_file:
        imagem_referencia = carregar_imagem_upload(camera_file)

# --- ÁREA PRINCIPAL (FOTOS DO EVENTO) ---
st.header("2. Fotos do Evento")
arquivos_evento = st.file_uploader(
    "Selecione TODAS as fotos do evento (pode selecionar várias de uma vez)", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

# --- BOTÃO DE AÇÃO ---
if st.button("🚀 INICIAR BUSCA", type="primary"):
    if imagem_referencia is None:
        st.error("⚠️ Você precisa enviar sua foto de referência primeiro (na barra lateral)!")
    elif not arquivos_evento:
        st.error("⚠️ Você precisa selecionar as fotos do evento para pesquisar!")
    else:
        # Processamento
        with st.spinner('⏳ Analisando fotos... Isso pode demorar um pouco.'):
            
            # 1. Codificar o rosto de referência
            try:
                ref_encodings = face_recognition.face_encodings(imagem_referencia)
                if not ref_encodings:
                    st.error("❌ Não encontrei nenhum rosto na sua foto de referência. Tente uma foto mais nítida.")
                    st.stop()
                
                seu_rosto = ref_encodings[0]
                
                # Listas para guardar resultados
                fotos_encontradas = []
                nomes_encontrados = []
                
                # Barra de progresso visual
                progresso = st.progress(0)
                total = len(arquivos_evento)
                
                for i, arquivo in enumerate(arquivos_evento):
                    # Atualiza barra
                    progresso.progress((i + 1) / total)
                    
                    try:
                        # Carrega imagem do evento
                        img_evento = carregar_imagem_upload(arquivo)
                        
                        # Busca rostos
                        rostos_evento = face_recognition.face_encodings(img_evento)
                        
                        if rostos_evento:
                            # Compara (Tolerância 0.6 para pegar fotos difíceis)
                            match = face_recognition.compare_faces(rostos_evento, seu_rosto, tolerance=0.6)
                            
                            if True in match:
                                fotos_encontradas.append(img_evento)
                                nomes_encontrados.append(arquivo.name)
                                
                    except Exception as e:
                        print(f"Erro ao ler arquivo: {e}")

                # --- RESULTADO FINAL ---
                st.success(f"🎉 Processo finalizado! Encontrei você em {len(fotos_encontradas)} fotos.")
                
                if fotos_encontradas:
                    # Mostrar galeria
                    st.subheader("Galeria de Fotos Encontradas")
                    cols = st.columns(3) # Grade de 3 colunas
                    for idx, foto in enumerate(fotos_encontradas):
                        cols[idx % 3].image(foto, caption=nomes_encontrados[idx], use_container_width=True)

                    # Criar ZIP para download
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for nome, img_array in zip(nomes_encontrados, fotos_encontradas):
                            # Converter numpy array de volta para bytes para salvar no zip
                            img_pil = Image.fromarray(img_array)
                            img_byte_arr = io.BytesIO()
                            img_pil.save(img_byte_arr, format='JPEG')
                            zf.writestr(nome, img_byte_arr.getvalue())
                    
                    st.download_button(
                        label="⬇️ BAIXAR TODAS AS FOTOS (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="minhas_fotos_encontradas.zip",
                        mime="application/zip",
                        type="primary"
                    )
                else:
                    st.warning("Poxa, não encontrei você em nenhuma foto. Tente mudar sua foto de referência.")
                    
            except Exception as e:
                st.error(f"Erro crítico: {e}")