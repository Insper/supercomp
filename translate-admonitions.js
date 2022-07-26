
// Tradução dos admonitions
let en_pt = {
    "Note": "Aviso",
    "Tip": "Dica",
    "Question": "Pergunta",
    "Warning": "Cuidado",
    "Example": "Tarefa",
    "Important": "Importante"
};

let admonitions_to_translate = document.querySelectorAll(".admonition-title");
for (var i = 0; i < admonitions_to_translate.length; i++) {
    let el = admonitions_to_translate[i];
    
    if (el.innerText in en_pt) {
        el.innerText = en_pt[el.innerText];
    }
}


/* Contagem automática de tarefas e perguntas */
let counters = ["example", "question"];
counters.forEach(function (admo_type) {
    let elements = document.querySelectorAll(".admonition." + admo_type + " > .admonition-title");
    for (let i = 0; i < elements.length; i++) {
        let el = elements[i];
        el.innerText += " " + (i+1);
    }    
});


// Adicionar embed dos slides se existir
let nome_aula = window.location.href;
nome_aula = nome_aula.split("/");
nome_aula = nome_aula[nome_aula.length - 2];
nome_aula = site_url + "aulas/" + nome_aula + "/slides.pdf";
let xhr = new XMLHttpRequest();
xhr.addEventListener("load", function(ev) {
    if (xhr.status == 200) {       
        let emb = document.createElement("embed");
        emb.src = nome_aula;
        emb.width = 500;
        emb.height = 375;
        emb.type = "application/pdf";
        
        let div_center = document.createElement("div");
        div_center.style = "text-align: center";
        div_center.className = "no-print";
        div_center.appendChild(emb);
        
        document.querySelector("article.md-typeset h1").insertAdjacentElement("afterend", div_center);
    }
});
xhr.open("head", nome_aula);
xhr.send();
 