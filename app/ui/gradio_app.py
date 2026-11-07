"""PolyglotSound artist studio (Gradio). Run: python -m app.ui.gradio_app"""
from __future__ import annotations

import os

import gradio as gr

from app.ui import handlers as H
from app.ui.render import CSS
from engines.translation.llm_adapter import ollama_available

CONSENT_TEXT = ("Je soussigne(e), artiste, autorise PolyglotSound a utiliser les extraits de ma voix que je fournis "
                "pour creer et appliquer un modele de mon timbre vocal, uniquement pour adapter mes propres oeuvres. "
                "Je peux retirer ce consentement a tout moment.")


def build() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="orange", secondary_hue="violet", neutral_hue="slate", text_size="lg",
                           font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"])
    with gr.Blocks(title="PolyglotSound Studio", theme=theme, css=CSS) as demo:
        pid = gr.State("")
        models = ollama_available()
        with gr.Column(elem_id="ps-hero"):
            gr.Markdown("# PolyglotSound\nFaites voyager votre chanson dans une autre langue, sur votre melodie, "
                        "avec votre voix. L'original n'est jamais modifie.")
        gr.Markdown("**Comment faire :** ① onglet **1 · Morceau** : importez un audio ou une video, choisissez les "
                    "langues, cliquez sur **Analyser**. ② En bas de la page : la traduction, puis la **version traduite "
                    "chantee** a ecouter. ③ Pour un texte seul : onglet **Traduction des paroles**.", elem_id="ps-steps")
        with gr.Row():
            artist = gr.Textbox(label="Nom de l'artiste", value="", scale=2)
            title = gr.Textbox(label="Titre du morceau", scale=2)

        with gr.Tab("Traduction des paroles"):
            gr.Markdown("Collez vos paroles (une ligne chantee par ligne). Pour chaque ligne : la **traduction fidele**, "
                        "puis la **version chantable** qui garde le nombre de syllabes. Environ 3 s par ligne.")
            with gr.Row():
                t_src = gr.Dropdown(H.LANG_CHOICES, value="fra", label="Langue des paroles")
                t_tgt = gr.Dropdown(H.LANG_CHOICES, value="eng", label="Traduire en")
                t_llm = gr.Checkbox(value=False, interactive=bool(models),
                                    label="IA locale pour des versions plus chantables (plus lent)")
            t_text = gr.Textbox(lines=8, label="Paroles", placeholder="Je chante pour toi ce soir / Mon coeur bat au rythme de ta voix")
            t_go = gr.Button("Traduire et ecouter", variant="primary", size="lg", elem_classes="ps-big")
            t_table = gr.Dataframe(headers=["#", "Original", "Traduction fidele", "Version chantable",
                                            "Syllabes (notes -> version)", "Score /100"], wrap=True, interactive=False)
            t_file = gr.File(label="Telecharger la traduction (.txt)")
            with gr.Accordion("Les 3 propositions par ligne et leurs scores detailles", open=False):
                t_cards = gr.HTML()
            t_audio_msg = gr.Markdown()
            t_audio = gr.Audio(label="Ecouter la traduction (voix de synthese)", interactive=False)

        with gr.Tab("1 · Morceau"):
            with gr.Row():
                file = gr.File(label="Votre morceau : audio (MP3, WAV, FLAC, M4A...) ou video (MP4, MOV, MKV, WEBM)",
                               file_types=["audio", "video", ".mp4", ".mov", ".mkv", ".webm", ".m4a"], type="filepath")
                with gr.Column():
                    src = gr.Dropdown(H.LANG_CHOICES, value="fon", label="Langue chantee")
                    tgt = gr.Dropdown(H.LANG_CHOICES, value="fra", label="Langue de la nouvelle version")
                    with gr.Row():
                        start_s = gr.Number(value=float(os.environ.get("POLYGLOT_EXCERPT_START", 0)),
                                            label="Debut de l'extrait (s)", minimum=0)
                        dur_s = gr.Number(value=float(os.environ.get("POLYGLOT_EXCERPT_DUR", 20)),
                                          label="Duree de l'extrait (s), 0 = tout le morceau", minimum=0)
                    gr.Markdown("Conseil : traitez d'abord un refrain de 20 a 40 s (sur CPU, un morceau "
                                "complet prend 15 a 30 min).")
                    go = gr.Button("Analyser le morceau", variant="primary", size="lg", elem_classes="ps-big")
            info = gr.Markdown()
            with gr.Row():
                voc = gr.Audio(label="Voix isolee", interactive=False)
                inst = gr.Audio(label="Instrumental (conserve tel quel)", interactive=False)
            fig = gr.Plot(label="Forme d'onde, temps forts et melodie")
            gr.Markdown("### Proposition dans la nouvelle langue")
            a_lyrics = gr.Textbox(lines=6, label="Paroles detectees (corrigez-les ou tapez-les si besoin)")
            a_go = gr.Button("Traduire ces paroles", variant="primary")
            a_msg = gr.Markdown()
            a_table = gr.Dataframe(headers=["#", "Original", "Traduction fidele", "Version chantable",
                                            "Syllabes (notes -> version)", "Score /100"], wrap=True, interactive=False)
            a_file = gr.File(label="Telecharger la traduction (.txt)")
            gr.Markdown("### Ecouter la version traduite")
            a_song_msg = gr.Markdown("La version chantee se cree automatiquement apres la traduction (1 a 4 min).")
            a_song = gr.Audio(label="Nouvelle version chantee dans la langue choisie", interactive=False)
            a_song_files = gr.File(label="Telecharger la version chantee (WAV + MP3)", file_count="multiple")
            with gr.Accordion("Les 3 propositions par ligne et leurs scores", open=False):
                a_cards = gr.HTML()

        with gr.Tab("2 · Paroles"):
            hint = gr.Markdown()
            lyrics = gr.Textbox(label="Paroles (une ligne chantee par ligne) : corrigez ou saisissez-les", lines=8)
            own_voice = gr.Checkbox(value=False, label=("Chanter avec la voix d'origine du morceau : je declare que "
                                    "c'est MA voix et j'autorise sa conversion (Seed-VC, enregistre au registre)"))
            with gr.Row():
                ok_lyr = gr.Button("Verifier les syllabes seulement")
                quick = gr.Button("Creer la nouvelle chanson", variant="primary", size="lg")
            quick_msg = gr.Markdown()
            with gr.Row():
                q_orig = gr.Audio(label="Original (extrait)", interactive=False)
                q_new = gr.Audio(label="Nouvelle chanson (voix de synthese, experimental)", interactive=False)
            q_files = gr.File(label="Telecharger la nouvelle chanson", file_count="multiple")
            syl_table = gr.Dataframe(headers=["#", "Ligne", "Notes", "Syllabes", "Rime", "Precision du rythme"],
                                     interactive=False, wrap=True)
            fig2 = gr.Plot(label="Syllabes placees sur la melodie (orange = temps fort)")

        with gr.Tab("3 · Adaptations"):
            with gr.Row():
                use_llm = gr.Checkbox(value=bool(models), label="Propositions artistiques par IA locale (Ollama)")
                llm = gr.Dropdown(models, value=("llama3.2:latest" if "llama3.2:latest" in models else
                                                 (models[0] if models else None)), label="Modele local")
                gen = gr.Button("Proposer 3 adaptations par ligne", variant="primary")
            cards = gr.HTML()
            gr.Markdown("**Votre choix** : remplacez le texte par l'option preferee ou par vos propres mots.")
            choice = gr.Dataframe(headers=["#", "Original", "Adaptation retenue"], interactive=True, wrap=True)
            with gr.Row():
                save_choice = gr.Button("Valider mes choix")
                relisten = gr.Button("Reecouter avec mes choix", variant="primary")
            choice_msg = gr.Markdown()
            with gr.Accordion("Journal technique", open=False):
                logs = gr.Textbox(lines=6, show_label=False)

        with gr.Tab("4 · Voix"):
            gr.Markdown("### Consentement de l'artiste\nAucune voix n'est clonee sans l'accord ecrit de son proprietaire.")
            statement = gr.Textbox(value=CONSENT_TEXT, lines=3, label="Declaration")
            agree = gr.Checkbox(label="Je suis l'artiste (ou son representant autorise) et j'accepte")
            consent_btn = gr.Button("Enregistrer le consentement")
            consent_msg = gr.Markdown()
            mode = gr.Radio(["Studio : je chante une voix guide dans la nouvelle langue",
                             "Auto (experimental) : voix guide synthetisee"], value=None, label="Mode")
            with gr.Row():
                ref = gr.Audio(label="Reference de votre voix (5 a 30 s, voix seule)", type="filepath")
                guide = gr.Audio(label="Voix guide (mode Studio)", type="filepath")
            steps = gr.Slider(10, 50, value=25, step=5, label="Qualite (etapes de diffusion, plus = plus lent)")
            conv = gr.Button("Generer la voix", variant="primary")
            conv_out = gr.Audio(label="Nouvelle voix", interactive=False)
            conv_path = gr.State("")
            conv_msg = gr.Markdown()

        with gr.Tab("5 · Version finale"):
            gain = gr.Slider(-6, 6, value=0, step=0.5, label="Niveau de la voix (dB)")
            fin = gr.Button("Mixer, signer (filigrane) et exporter", variant="primary")
            with gr.Row():
                a_orig = gr.Audio(label="Original", interactive=False)
                a_new = gr.Audio(label="Nouvelle version", interactive=False)
            files = gr.File(label="Telecharger", file_count="multiple")
            prov = gr.Code(label="Provenance (registre + filigrane)", language="json")

        t_go.click(H.translate_text, [t_text, t_src, t_tgt, t_llm, llm], [t_table, t_cards, t_file]).then(
            H.translation_audio, [t_table, t_tgt], [t_audio, t_audio_msg])
        go.click(H.analyse, [file, src, tgt, start_s, dur_s], [pid, voc, inst, info, hint, lyrics, fig]).then(
            lambda t: t, [lyrics], [a_lyrics]).then(
            H.auto_translate, [a_lyrics, src, tgt], [a_table, a_cards, a_file, a_msg]).then(
            H.auto_song, [pid, a_lyrics, artist, title], [a_song, a_song_files, a_song_msg])
        a_go.click(H.auto_translate, [a_lyrics, src, tgt], [a_table, a_cards, a_file, a_msg]).then(
            lambda t: t, [a_lyrics], [lyrics]).then(
            H.auto_song, [pid, a_lyrics, artist, title], [a_song, a_song_files, a_song_msg])
        quick.click(H.quick_song, [pid, lyrics, artist, title, use_llm, llm, own_voice],
                    [syl_table, fig2, cards, choice, q_orig, q_new, q_files, quick_msg])
        ok_lyr.click(H.validate_lyrics, [pid, lyrics, artist], [syl_table, fig2])
        gen.click(H.adapt, [pid, use_llm, llm], [cards, choice, logs])
        save_choice.click(H.choose, [pid, choice, artist], [choice_msg])
        relisten.click(H.preview_choice, [pid, choice, artist, title], [q_orig, q_new, q_files, choice_msg])
        consent_btn.click(H.grant, [artist, statement, agree], [consent_msg])
        conv.click(H.convert, [pid, artist, mode, ref, guide, steps], [conv_out, conv_path, conv_msg])
        fin.click(H.finalize, [pid, artist, title, mode, conv_path, gain], [a_orig, a_new, files, prov])
    return demo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="lien public gradio.live (Colab)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=False)
