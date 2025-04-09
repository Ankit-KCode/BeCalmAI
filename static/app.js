const openBtn = document.querySelector('.chatbox__btn');
const chatBox = document.querySelector('.chatbox__support');
const sendBtn = document.querySelector('.send__btn');
const input = document.querySelector('input');
const chatMessages = document.querySelector('.chatbox__messages');
const greetings = document.querySelectorAll('.messages__greeting');

let state = false;

const toggleState = () => {
    state = !state;
    chatBox.classList.toggle('chatbox--active');
    greetings.forEach(greeting => greeting.classList.toggle('active'));
};

const display = () => {
    openBtn.addEventListener('click', toggleState);
    sendBtn.addEventListener('click', sendMessage);

    input.addEventListener("keyup", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
};

const sendMessage = () => {
    const userInput = input.value.trim();
    if (!userInput) return;

    addMessage('operator', userInput);
    input.value = "";

    // Loading animation
    const botDiv = addMessage('visitor', `
        <div class="loader">
            <div class="loader__dot"></div>
            <div class="loader__dot"></div>
            <div class="loader__dot"></div>
        </div>
    `);

    fetch('http://127.0.0.1:5050/predict', {
        method: 'POST',
        body: JSON.stringify({ message: userInput }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(r => r.json())
    .then(r => {
        const botResponse = r.answer || "Sorry, I didn't understand that.";
        setTimeout(() => {
            botDiv.innerText = botResponse;
            scrollToBottom();
        }, 1200);
    })
    .catch(err => {
        console.error("API Error:", err);
        setTimeout(() => {
            botDiv.innerText = "Oops! Something went wrong. Please try again.";
            scrollToBottom();
        }, 1200);
    });
};

const addMessage = (sender, text) => {
    const msgDiv = document.createElement("div");
    msgDiv.className = `messages__item messages__item--${sender}`;
    msgDiv.innerHTML = text;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
};

const scrollToBottom = () => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

display();
