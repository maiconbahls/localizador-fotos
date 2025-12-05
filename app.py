import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import zipfile
import io

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Localizador de Fotos", page_icon="📸", layout="wide")

st.title("📸 Localizador de Fotos com IA")
st.markdown("### Encontre suas fotos automaticamente")
st.markdown("Faça upload da sua foto de referência e das fotos do evento. O sistema vai separar tudo para você.")

# --- FUNÇÃO DE AJUDA ---
def carregar_imagem(upload_file):
    """Converte o arquivo enviado pelo site em uma imagem que a IA entende"""
    if upload_file is not None:
        image = Image.open(upload_file)
        image = image.convert('RGB')
        return np.array(image)
    return None

# --- BARRA LATERAL (SUA FOTO) ---
st.sidebar.header("1. Quem procurar?")
opcao = st.sidebar.radio("Escolha a fonte:", ["📁 Enviar Arquivo", "📷 Usar Webcam"])

imagem_referencia = None

if opcao == "📁 Enviar Arquivo":
    ref_file = st.sidebar.file_uploader("Sua foto de rosto", type=['jpg', 'png', 'jpeg'])
    if ref_file:
        imagem_referencia = carregar_imagem(ref_file)
        st.sidebar.image(imagem_referencia, caption="Referência", use_container_width=True)

elif opcao == "📷 Usar Webcam":
    camera_file = st.sidebar.camera_input("Tire uma foto")
    if camera_file:
        imagem_referencia = carregar_imagem(camera_file)

# --- ÁREA PRINCIPAL (FOTOS DO EVENTO) ---
st.header("2. Fotos do Evento")
arquivos_evento = st.file_uploader(
    "Arraste e solte TODAS as fotos do evento aqui", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

# --- BOTÃO DE PROCESSAMENTO ---
if st.button("🚀 INICIAR BUSCA", type="primary"):
    
    # Validações antes de começar
    if imagem_referencia is None:
        st.error("⚠️ Você precisa definir sua foto de referência (na barra lateral)!")
    elif not arquivos_evento:
        st.error("⚠️ Você precisa selecionar as fotos do evento para pesquisar!")
    else:
        # Começa o processamento
        with st.spinner('⏳ Analisando fotos... Isso pode levar alguns minutos.'):
            
            try:
                # 1. Ler o rosto de referência
                ref_encodings = face_recognition.face_encodings(imagem_referencia)
                
                if not ref_encodings:
                    st.error("❌ Não encontrei rosto na sua foto de referência. Tente uma foto mais nítida e de frente.")
                else:
                    seu_rosto = ref_encodings[0]
                    
                    fotos_encontradas = []
                    nomes_encontrados = []
                    
                    # Barra de progresso
                    progresso = st.progress(0)
                    total_fotos = len(arquivos_evento)

                    # 2. Loop por todas as fotos do evento
                    for i, arquivo in enumerate(arquivos_evento):
                        # Atualiza barra de progresso
                        progresso.progress((i + 1) / total_fotos)
                        
                        try:
                            img_evento = carregar_imagem(arquivo)
                            
                            # Busca rostos na foto do evento
                            rostos_evento = face_recognition.face_encodings(img_evento)
                            
                            if rostos_evento:
                                # Compara com o seu rosto (Tolerância 0.6)
                                match = face_recognition.compare_faces(rostos_evento, seu_rosto, tolerance=0.6)
                                
                                if True in match:
                                    fotos_encontradas.append(img_evento)
                                    nomes_encontrados.append(arquivo.name)
                                    
                        except Exception as e:
                            # Ignora erros de leitura e continua
                            pass

                    # --- RESULTADOS ---
                    if len(fotos_encontradas) > 0:
                        st.success(f"🎉 Sucesso! Encontrei você em {len(fotos_encontradas)} fotos.")
                        
                        # Mostra Galeria
                        st.subheader("Galeria")
                        cols = st.columns(3)
                        for idx, foto in enumerate(fotos_encontradas):
                            cols[idx % 3].image(foto, caption=nomes_encontrados[idx], use_container_width=True)

                        # --- CRIA O ZIP PARA DOWNLOAD ---
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            for nome, img_array in zip(nomes_encontrados, fotos_encontradas):
                                # Converte de volta para bytes para salvar no zip
                                img_pil = Image.fromarray(img_array)
                                img_byte_arr = io.BytesIO()
                                img_pil.save(img_byte_arr, format='JPEG')
                                zf.writestr(nome, img_byte_arr.getvalue())
                        
                        st.download_button(
                            label="⬇️ BAIXAR TUDO (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="minhas_fotos.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    else:
                        st.warning("Poxa, não encontrei você em nenhuma foto. Tente mudar sua foto de referência.")
            
            except Exception as e:
                st.error(f"Erro técnico: {e}")
