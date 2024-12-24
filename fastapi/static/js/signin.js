        document.getElementById("login-submit").addEventListener("click", function() {
            document.getElementById('emailError').textContent = '';
    document.getElementById('passwordError').textContent = '';
    const successMessageContainer = document.getElementById('successMessage');
    successMessageContainer.textContent = ''; // Clear previous success message

    // Retrieve form values

    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value.trim();

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

        });