import streamlit as st
from src.backend import process_pdf, create_vector_db, load_rag_chain

def render():
    st.title("RAG sur PDF — Llama 3.1 8B")

    # Initialisation de l'historique de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Upload PDF ---
    uploaded_file = st.file_uploader("Téléchargez un fichier PDF de max 200 Mo", type=["pdf"])

    if uploaded_file is not None:
        # Vérification de la taille
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("Le fichier dépasse la taille maximale autorisée de 200 Mo.")
        else:
            # On vérifie si ce fichier a déjà été traité pour éviter de tout refaire à chaque action
            if st.session_state.get("processed_file") != uploaded_file.name:
                pdf_path = f"./{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
        
                with st.spinner("Traitement de votre document PDF..."):
                    chunks = process_pdf(pdf_path)
                    if chunks:
                        create_vector_db(chunks)
                        st.session_state.chain = load_rag_chain()
                        st.session_state.processed_file = uploaded_file.name
                        # On réinitialise l'historique quand un nouveau fichier est chargé
                        st.session_state.messages = []
                        st.success("PDF traité et chaîne RAG chargée avec succès !")
                    else:
                        st.error("Erreur lors du traitement du PDF.")
    else:
        st.info("Veuillez télécharger un PDF pour activer le chat.")

    # --- Affichage de l'historique des messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Affichage des sources s'il y en a
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources utilisées"):
                    for src in msg["sources"]:
                        st.write(src)

    # --- Zone de question ---
    is_ready = "chain" in st.session_state

    # chat_input sera grisé tant que le document n'est pas chargé et traité
    question = st.chat_input("Posez votre question sur le document :", disabled=not is_ready)

    if question:
        # Ajouter et afficher le message utilisateur immédiatement
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Générer et afficher la réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse..."):
                try:
                    result = st.session_state.chain.invoke({"query": question})
                    rep = result["result"]
                    st.markdown(rep)
                    
                    # On stocke et affiche les sources
                    sources_list = []
                    for doc in result.get("source_documents", []):
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:150] + "..."
                        sources_list.append(f"- **Page {page}** : {snippet}")

                    if sources_list:
                        with st.expander("Sources utilisées"):
                            for src in sources_list:
                                st.write(src)

                    # Sauvegarde dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": rep,
                        "sources": sources_list
                    })

                except Exception as e:
                    st.error(f"Erreur lors de la génération : {str(e)}")
