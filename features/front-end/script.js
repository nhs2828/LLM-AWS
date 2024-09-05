const chatInput = 
    document.querySelector('.user-input textarea');
const sendChatBtn = 
    document.querySelector('.user-input button');
const chatbox = document.querySelector(".box");

// FOR DEV
// const EC2_ENDPOINT = "http://localhost:9999/rep"

const EC2_ENDPOINT = "EC2-public-IP4:9999/rep"

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = 
        className === "user-mess" ? `<p>${message}</p>` : `<p>${message}</p>`;
    chatLi.innerHTML = chatContent;
    return chatLi;
}

const generateResponse = (incomingChatLi) => {
    const messageElement = incomingChatLi
    .querySelector("p");
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            'question': userMessage
        })
    };
 
    fetch(EC2_ENDPOINT, requestOptions)
        .then(res => {
            if (!res.ok) {
                throw new Error("Network response was not ok");
            }
            return res.json();
        })
        // .then(reponse => reponse.json())
        .then(data => {
            messageElement
            .textContent = data.rep;
        })
        // .catch((error) => {
        //     messageElement
        //     .classList.add("error");
        //     messageElement
        //     .textContent = "Oops! Something went wrong. Please try again!";
        // })
        .finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
    document.getElementById("sendQuestion").disabled = false;
};

const handleChat = () => {
    userMessage = chatInput.value.trim();
    if (!userMessage) {
        return;
    }
    document.getElementById("sendQuestion").disabled = true;
    chatbox
    .appendChild(createChatLi(userMessage, "user-mess"));
    chatbox
    .scrollTo(0, chatbox.scrollHeight);
 
    setTimeout(() => {
        const incomingChatLi = createChatLi("Thinking...", "bot-rep")
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        generateResponse(incomingChatLi);
    }, 600);
}
 
sendChatBtn.addEventListener("click", handleChat);