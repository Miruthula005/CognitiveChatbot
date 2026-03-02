// ---------------- QUIZ SCRIPT ----------------
let currentStyle = "";
let totalQuestions = 8;
let currentQuestionIndex = 0;

function loadNext() {
  fetch("/next_question")
    .then((response) => response.json())
    .then((data) => {
      if (data.done) {
        document.getElementById("question").innerHTML = "Quiz Finished!";
        document.getElementById("result").innerHTML =
          "Learning Style: " + data.dominant;
        document.querySelector(".options").style.display = "none";
        updateProgress(totalQuestions); // ensure progress reaches 100%

        // Show Next button to go to after_result page
        const nextBtn = document.createElement("button");
        nextBtn.innerText = "Next";
        nextBtn.classList.add("option");
        nextBtn.style.marginTop = "20px";
        nextBtn.onclick = function () {
          window.location.href = "/after_result";
        };
        document.querySelector(".container").appendChild(nextBtn);
      } else {
        // Animate question
        document.getElementById("question").classList.remove("show");
        setTimeout(() => {
          document.getElementById("question").innerHTML = data.question;
          currentStyle = data.style;
          document.getElementById("question").classList.add("show");
        }, 200);

        // Update progress
        currentQuestionIndex++;
        updateProgress(currentQuestionIndex);
      }
    });
}

function sendAnswer(answer) {
  fetch("/submit_answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer: answer, style: currentStyle }),
  }).then(() => {
    loadNext();
  });
}

function updateProgress(questionIndex) {
  let progressPercent = (questionIndex / totalQuestions) * 100;
  document.getElementById("progress").style.width = progressPercent + "%";
}

// Load first question
loadNext();
