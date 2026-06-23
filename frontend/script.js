document.getElementById("form").addEventListener("submit", async function(e) {
    e.preventDefault();

    const enrolled = document.getElementById("enrolled").files[0];
    const test = document.getElementById("test").files[0];

    const formData = new FormData();
    formData.append("enrolled", enrolled);
    formData.append("test", test);

    const res = await fetch("http://127.0.0.1:8000/verify", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    const resultBox = document.getElementById("resultBox");
    const resultText = document.getElementById("resultText");
    const distanceText = document.getElementById("distanceText");

    resultBox.classList.remove("hidden");

    resultText.innerText = "Result: " + data.result;
    distanceText.innerText = "Distance: " + data.distance.toFixed(4);
});