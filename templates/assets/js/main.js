var method = 2;
const temp = {
    "Rule=based Machine Translation": '',
    "Statistical Machine Translation": '',
    "Neural Machine Translation": ''
}

const inputText = document.getElementById("source_text");
var modelBtn = document.querySelectorAll('.model');
for (let i = 0; i < modelBtn.length; ++i){
    modelBtn[i].addEventListener('click', function() {
        
        const grammarArea = document.getElementById('source_grammar')
        const checkActive = document.querySelector('.active');

        // REMOVE ACTIVE OLD MODEL
        checkActive.classList.remove('active')
        if (checkActive.innerText == "Rule-based Machine Translation") {
            grammarArea.style.display = "none";
            document.querySelector('textarea').style.minHeight = "100px";
        }
        temp[checkActive.innerText] = document.querySelector('textarea').value


        // CHOOSE ACTIVE NEW MODEL
        modelBtn[i].classList.add('active')
        if (modelBtn[i].innerText == "Rule-based Machine Translation"){
            method = 0;
            grammarArea.style.display = "inline-block";
            document.querySelector('textarea').style.minHeight = "200px";
        }
        else if (modelBtn[i].innerText == "Statistical Machine Translation") {
            method = 1;
        }
        else method = 2;

        if (temp[modelBtn[i].innerText] === undefined) {
            // console.log("NOT ALLOW");
            document.querySelector('textarea').value = ''
        }
        else document.querySelector('textarea').value = temp[modelBtn[i].innerText]
    })
}


const languageOption = document.getElementById('target_language');
languageOption.addEventListener('change', function() {
    console.log(languageOption.value);
})

const grammarInput = document.getElementById('source_grammar')



// TRANSLATE BUTTON
const submitTrans = document.getElementById('submitTrans');
submitTrans.addEventListener('click', function() {
    console.log("OKK Click")
    console.log(`
        SENTENCE: ${inputText.value},
        GRAMMAR: ${grammarInput.value},
        METHOD: ${method}
    `)
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sentence: inputText.value,
            grammar: grammarInput.value,
            method: method,
        }),
    })
    .then(response => response.json())
    .then(data => {
        const outputElement = document.getElementById("target_text");
        console.log(outputElement)
        outputElement.innerText = data.output       
    })
    .catch((error) => {
        const outputElement = document.getElementById("target_text");        
        outputElement.innerText = "Error"      
        
    })
})
