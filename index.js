
    // Theme Toggle
    function toggleTheme() {
      const body = document.body;
      const themeIcon = document.getElementById('theme-icon');
      
      if (body.getAttribute('data-theme') === 'dark') {
        body.removeAttribute('data-theme');
        themeIcon.className = 'fas fa-moon';
        localStorage.setItem('theme', 'light');
      } else {
        body.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-sun';
        localStorage.setItem('theme', 'dark');
      }
    }

    // Load saved theme
    function loadTheme() {
      const savedTheme = localStorage.getItem('theme');
      const themeIcon = document.getElementById('theme-icon');
      
      if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-sun';
      }
    }

    // Mobile Menu Toggle
    function toggleMobileMenu() {
      const mobileMenu = document.getElementById('mobileMenu');
      mobileMenu.classList.toggle('active');
      
      // Prevent body scroll when menu is open
      if (mobileMenu.classList.contains('active')) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    }

    // Animated Particles
    function createParticles() {
      const particlesContainer = document.getElementById('particles');
      const particleCount = window.innerWidth > 768 ? 50 : 20;
      
      for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
        particle.style.animationDelay = Math.random() * 5 + 's';
        particlesContainer.appendChild(particle);
      }
    }

    // Smooth Scroll
    function smoothScroll(target) {
      document.querySelector(target).scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }

    // Update navigation links to use smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = this.getAttribute('href');
        smoothScroll(target);
      });
    });

    // Header Scroll Effect
    const header = document.querySelector("header");
    const backToTop = document.getElementById("backToTop");
    
    window.addEventListener("scroll", () => {
      const scrollY = window.scrollY;
      
      // Header effect
      header.classList.toggle("scrolled", scrollY > 50);
      
      // Back to top button
      if (scrollY > 300) {
        backToTop.classList.add("visible");
      } else {
        backToTop.classList.remove("visible");
      }
    });

    // Back to Top Function
    function scrollToTop() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }

    // Intersection Observer for Animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          
          // Trigger counter animation for stats
          if (entry.target.classList.contains('stat')) {
            animateCounter(entry.target);
          }
        }
      });
    }, observerOptions);

    // Observe all animated elements
    document.querySelectorAll("[data-animate]").forEach(el => {
      observer.observe(el);
    });

    // Counter Animation
    function animateCounter(statElement) {
      const counter = statElement.querySelector('.counter');
      if (!counter || counter.classList.contains('animated')) return;
      
      counter.classList.add('animated');
      const target = parseInt(counter.getAttribute('data-target'));
      const increment = target / 100;
      let current = 0;
      
      const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
          current = target;
          clearInterval(timer);
        }
        counter.textContent = Math.floor(current).toLocaleString();
      }, 20);
    }

    // Testimonial Carousel
    let currentTestimonial = 0;
    const testimonials = document.querySelectorAll('.testimonial');
    const totalTestimonials = testimonials.length;
    let autoSlideInterval;

    function updateCarousel() {
      const carousel = document.getElementById('testimonialCarousel');
      const translateX = -currentTestimonial * (testimonials[0].offsetWidth + 16); // 16px for gap
      carousel.style.transform = `translateX(${translateX}px)`;
      
      // Update dots
      updateDots();
    }

    function createDots() {
      const dotsContainer = document.getElementById('carouselDots');
      dotsContainer.innerHTML = '';
      
      for (let i = 0; i < totalTestimonials; i++) {
        const dot = document.createElement('div');
        dot.className = 'dot';
        if (i === 0) dot.classList.add('active');
        dot.addEventListener('click', () => goToTestimonial(i));
        dotsContainer.appendChild(dot);
      }
    }

    function updateDots() {
      const dots = document.querySelectorAll('.dot');
      dots.forEach((dot, index) => {
        dot.classList.toggle('active', index === currentTestimonial);
      });
    }

    function nextTestimonial() {
      currentTestimonial = (currentTestimonial + 1) % totalTestimonials;
      updateCarousel();
      resetAutoSlide();
    }

    function previousTestimonial() {
      currentTestimonial = (currentTestimonial - 1 + totalTestimonials) % totalTestimonials;
      updateCarousel();
      resetAutoSlide();
    }

    function goToTestimonial(index) {
      currentTestimonial = index;
      updateCarousel();
      resetAutoSlide();
    }

    function startAutoSlide() {
      autoSlideInterval = setInterval(nextTestimonial, 5000);
    }

    function resetAutoSlide() {
      clearInterval(autoSlideInterval);
      startAutoSlide();
    }

    // Touch/Swipe Support for Testimonials
    let startX = 0;
    let currentX = 0;
    let isDragging = false;

    const carouselContainer = document.querySelector('.carousel-container');

    carouselContainer.addEventListener('touchstart', (e) => {
      startX = e.touches[0].clientX;
      isDragging = true;
      clearInterval(autoSlideInterval);
    });

    carouselContainer.addEventListener('touchmove', (e) => {
      if (!isDragging) return;
      e.preventDefault();
      currentX = e.touches[0].clientX;
    });

    carouselContainer.addEventListener('touchend', (e) => {
      if (!isDragging) return;
      isDragging = false;
      
      const diffX = startX - currentX;
      if (Math.abs(diffX) > 50) {
        if (diffX > 0) {
          nextTestimonial();
        } else {
          previousTestimonial();
        }
      }
      startAutoSlide();
    });

    // Form Submission
    function handleFormSubmit(event) {
      event.preventDefault();
      
      // Get form elements
      const form = event.target;
      const button = form.querySelector('button[type="submit"]');
      const originalText = button.innerHTML;
      
      // Show loading state
      button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
      button.disabled = true;
      
      // Simulate form submission
      setTimeout(() => {
        button.innerHTML = '<i class="fas fa-check"></i> Message Sent!';
        button.style.background = '#4CAF50';
        
        // Reset form
        form.reset();
        
        // Reset button after 3 seconds
        setTimeout(() => {
          button.innerHTML = originalText;
          button.disabled = false;
          button.style.background = '';
        }, 3000);
      }, 2000);
    }

    // Resource Cards Click Handler
    document.querySelectorAll('.resource').forEach(card => {
      card.addEventListener('click', function() {
        // Add ripple effect
        const ripple = document.createElement('div');
        ripple.style.cssText = `
          position: absolute;
          border-radius: 50%;
          background: rgba(255, 107, 53, 0.3);
          transform: scale(0);
          animation: ripple 0.6s linear;
          pointer-events: none;
        `;
        
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (event.clientX - rect.left - size / 2) + 'px';
        ripple.style.top = (event.clientY - rect.top - size / 2) + 'px';
        
        this.style.position = 'relative';
        this.appendChild(ripple);
        
        setTimeout(() => {
          ripple.remove();
        }, 600);
        
        // Scroll to contact section
        setTimeout(() => {
          smoothScroll('#contact');
        }, 300);
      });
    });

    // Add ripple animation CSS
    const style = document.createElement('style');
    style.textContent = `
      @keyframes ripple {
        to {
          transform: scale(4);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);

    // Initialize everything when page loads
    document.addEventListener('DOMContentLoaded', function() {
      loadTheme();
      createParticles();
      createDots();
      startAutoSlide();
      
      // Add stagger effect to cards
      const cards = document.querySelectorAll('.trust-card, .step, .resource');
      cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
      });
    });

    // Keyboard navigation for accessibility
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && document.getElementById('mobileMenu').classList.contains('active')) {
        toggleMobileMenu();
      }
    });

    // Resize handler
    window.addEventListener('resize', function() {
      updateCarousel();
      
      // Recreate particles on resize
      const particlesContainer = document.getElementById('particles');
      particlesContainer.innerHTML = '';
      createParticles();
    });

    // Performance optimization: Reduce motion for users who prefer it
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.documentElement.style.setProperty('--animation-duration', '0.01s');
    }
