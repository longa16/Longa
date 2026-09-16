import streamlit as st
from src.backend import load_rag_chain

def render():
    st.title("RAG sur PDF — Assistant Local")

    # Initialisation de l'historique de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chargement unique de la chaîne et de la BDD (qui doit déjà exister via ingest.py)
    if "chain" not in st.session_state:
        with st.spinner("Chargement de la base de connaissances..."):
            try:
                st.session_state.chain = load_rag_chain()
                st.success("Base chargée avec succès. L'assistant est prêt !")
            except Exception as e:
                st.error("Impossible de charger la base. Avez-vous lancé le script d'ingestion des données ?")
                st.stop()

    # --- Affichage de l'historique des messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources utilisées"):
                    for src in msg["sources"]:
                        st.write(src)

    # --- Zone de question ---
    question = st.chat_input("Posez votre question sur la documentation de l'entreprise :")

    if question:
        # Message utilisateur
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Réponse assistant
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents en cours..."):
                try:
                    result = st.session_state.chain.invoke({"query": question})
                    rep = result["result"]
                    st.markdown(rep)
                    
                    # Traitement des sources
                    sources_list = []
                    for doc in result.get("source_documents", []):
                        source = doc.metadata.get("source", "?")
                        # Formatage du nom du fichier pour l'affichage
                        filename = source.split("/")[-1].split("\\")[-1]
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:150] + "..."
                        sources_list.append(f"- **{filename} (Page {page})** : {snippet}")

                    if sources_list:
                        with st.expander("Sources utilisées"):
                            for src in sources_list:
                                st.write(src)

                    # Sauvegarde l'historique avec sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": rep,
                        "sources": sources_list
                    })

                except Exception as e:
                    st.error(f"Erreur lors de la génération : {str(e)}")
