document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const suggestions = document.querySelectorAll('.suggestion');

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage('user', userMessage);
            userInput.value = '';
            const botResponse = await getBotResponse(userMessage);
            addMessage('bot', botResponse);
        }
    });

    suggestions.forEach(button => {
        button.addEventListener('click', async () => {
            const userMessage = button.textContent;
            addMessage('user', userMessage);
            const botResponse = await getBotResponse(userMessage);
            addMessage('bot', botResponse);
        });
    });

    function addMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender);
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function getBotResponse(message) {
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error('Error fetching bot response:', error);
            return 'Error fetching response';
        }
    }
});
