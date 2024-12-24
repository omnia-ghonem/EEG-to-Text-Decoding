        document.getElementById("submit").addEventListener("click", function() {
            document.getElementById('emailError').textContent = '';
    document.getElementById('passwordError').textContent = '';
    const successMessageContainer = document.getElementById('successMessage');
    successMessageContainer.textContent = ''; // Clear previous success message

    // Retrieve form values
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value.trim();

    let isValid = true;

    // Name validation


    // Email validation
    if (!email) {
        document.getElementById('emailError').textContent = 'Email is required.';
        isValid = false;
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        document.getElementById('emailError').textContent = 'Enter a valid email address.';
        isValid = false;
    }

    // Password validation
    if (!password) {
        document.getElementById('messageError').textContent = 'Password is required.';
        isValid = false;
    }
            // Simple validation
    if (name && email && password) {
        // Save user credentials to localStorage
        const user = {
            name: name,
            email: email,
            password: password
        };
        localStorage.setItem("user", JSON.stringify(user));

        // Redirect to login page after successful sign-up
        window.location.href = "index.html";
    } else {
        alert("Please fill all fields.");
    }
    // Display success message if valid

    
        });