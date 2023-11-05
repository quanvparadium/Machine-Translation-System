const inputText = document.getElementById("source_text");
console.log(3)
console.log(inputText)
// console.log(inputText.value)
inputText.addEventListener("input", function() {
    const inputVal = inputText.value;
    const outputElement = document.getElementById("target-text");
    outputElement.textContent = inputVal;
})