const inputText = document.getElementById("source_text");
console.log(inputText)
// console.log(inputText.value)
// inputText.addEventListener("input", function() {
//     const inputVal = inputText.value;
//     const outputElement = document.getElementById("target-text");
//     outputElement.textContent = inputVal;
// })

var activeBtn = document.querySelectorAll('button.model');
console.log(activeBtn)
for (var i = 0; i < activeBtn.length; ++i)
activeBtn[i].addEventListener('click', function() {
    console.log(activeBtn[i]);
    activeBtn[i].classList.add('active');
})



const submitTrans = document.getElementById('submitTrans');
console.log(submitTrans)
submitTrans.addEventListener('click', function() {
    fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: inputText.value
        }),
    })
    .then(response => response.json())
    .then(data => {
        const outputElement = document.getElementById("target-text");
        outputElement.textContent = data.target_text       
    })
    .catch((error) => {
        const outputElement = document.getElementById("target-text");        
        outputElement.textContent = "Error"      
        
    })
})
