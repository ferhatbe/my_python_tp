import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent   # ← nouveau

# ── Configuration GitHub ───────────────────────────────────────────────────────
GITHUB_TOKEN  = os.environ["GITHUB_TOKEN"]
GITHUB_REPO   = os.environ["GITHUB_REPO"]
PR_NUMBER     = int(os.environ["GITHUB_PR_NUMBER"])
HEADERS       = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE_URL      = f"https://api.github.com/repos/{GITHUB_REPO}"


# ── Tools GitHub ───────────────────────────────────────────────────────────────
@tool
def get_pr_files(pr_number: int) -> str:
    """Récupère la liste des fichiers modifiés dans une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur GitHub {resp.status_code} : {resp.text}"
    files = resp.json()
    result = []
    for f in files:
        result.append(
            f"Fichier : {f['filename']}\n"
            f"Patch   :\n{f.get('patch', '(pas de patch)')}"
        )
    return "\n---\n".join(result)


@tool
def post_review_comment(pr_number: int, body: str) -> str:
    """Poste un commentaire général de revue sur une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    payload = {"body": body, "event": "COMMENT"}
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Commentaire posté (review_id={resp.json().get('id')})."
    return f"Erreur GitHub {resp.status_code} : {resp.text}"


@tool
def suggest_fix(pr_number: int, commit_id: str, path: str,
                line: int, suggestion: str, comment: str) -> str:
    """Poste une suggestion de correction inline sur une ligne précise de la PR.
    
    Args:
        pr_number  : numéro de la PR
        commit_id  : SHA du dernier commit de la PR (get_pr_info pour l'obtenir)
        path       : chemin du fichier (ex: calculator.py)
        line       : numéro de ligne dans le fichier
        suggestion : code de remplacement proposé (bloc ```suggestion```)
        comment    : explication de la suggestion
    """
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    body_text = f"{comment}\n\n```suggestion\n{suggestion}\n```"
    payload = {
        "commit_id": commit_id,
        "body": f"Suggestion sur {path}:{line}",   # body racine non vide (sinon 422)
        "event": "COMMENT",
        "comments": [
            {
                "path": path,
                "line": line,
                "body": body_text,
            }
        ]
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Suggestion inline postée sur {path}:{line}."
    return f"Erreur GitHub {resp.status_code} : {resp.text}"


@tool
def get_pr_info(pr_number: int) -> str:
    """Récupère les métadonnées d'une PR : HEAD commit SHA, branche, auteur."""
    url = f"{BASE_URL}/pulls/{pr_number}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur GitHub {resp.status_code} : {resp.text}"
    data = resp.json()
    return (
        f"Titre    : {data['title']}\n"
        f"Auteur   : {data['user']['login']}\n"
        f"Branche  : {data['head']['ref']}\n"
        f"Commit   : {data['head']['sha']}\n"
        f"Statut   : {data['state']}"
    )


# ── Modèle ─────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",
        "X-Title": "TP Acte 2"
    },
)

# ── Création de l'agent avec create_agent ────────────────────────────────────

tools = [
    get_pr_info,
    get_pr_files,
    post_review_comment,
    suggest_fix,
]

system_prompt = """
Tu es un reviewer senior spécialisé en revue de code GitHub.

Ta mission :
1. Récupère les informations de la Pull Request avec get_pr_info.
2. Analyse les fichiers modifiés avec get_pr_files.
3. Identifie les problèmes potentiels de qualité, bugs ou améliorations.
4. Poste obligatoirement un commentaire général avec post_review_comment.
5. Poste au moins une suggestion inline avec suggest_fix.

Pour une suggestion inline :
- récupère d'abord le commit SHA avec get_pr_info.
- utilise uniquement une ligne présente dans le diff GitHub.
- propose un remplacement complet dans le champ suggestion.
- explique le problème dans comment.

Tu dois réellement appeler les outils GitHub, pas seulement expliquer ce qu'il faudrait faire.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

# ── Invocation de l'agent ────────────────────────────────────────────────────

for event in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Analyse la Pull Request #{PR_NUMBER}. "
                    "Effectue une vraie revue GitHub avec commentaire "
                    "général et suggestion inline."
                ),
            }
        ]
    }
):
    print(event)