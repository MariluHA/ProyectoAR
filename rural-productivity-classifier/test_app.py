"""
Test script for Rural Productivity Classifier
Verifies that the web application works correctly using Playwright
"""

import subprocess
import time
import sys
import signal
from playwright.sync_api import sync_playwright

# Configuration
BASE_URL = "http://localhost:5000"
SERVER_STARTUP_TIME = 5  # seconds to wait for server startup

def start_flask_server():
    """Start the Flask server in background"""
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/workspace/rural-productivity-classifier"
    )
    return process

def test_page_load(playwright):
    """Test that the main page loads correctly"""
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    
    try:
        # Navigate to the page
        response = page.goto(BASE_URL, wait_until="networkidle")
        
        # Check if page loaded successfully
        assert response.status == 200, f"Page returned status {response.status}"
        print("✓ Page loaded successfully")
        
        # Check for main title
        title = page.title()
        assert "Productividad Rural" in title, f"Title mismatch: {title}"
        print("✓ Page title is correct")
        
        # Check for key sections
        header = page.locator("h1").first.text_content()
        assert "Clasificador" in header, f"Header not found: {header}"
        print("✓ Main header is present")
        
        browser.close()
        return True
        
    except Exception as e:
        print(f"✗ Page load test failed: {e}")
        browser.close()
        return False

def test_form_elements(playwright):
    """Test that all form elements are present"""
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    
    try:
        page.goto(BASE_URL, wait_until="networkidle")
        
        # Check for model selector
        model_select = page.locator("#modelo_ml")
        assert model_select.count() > 0, "Model selector not found"
        print("✓ Model selector is present")
        
        # Check for form input fields
        fields = [
            "#indice_org",
            "#nivel_educativo",
            "#pct_mujeres",
            "#pct_varones",
            "#tipo_producto",
            "#tiempo_ejecucion",
            "#brecha_territorial",
            "#precipitacion",
            "#temperatura",
            "#sequia"
        ]
        
        for field in fields:
            element = page.locator(field)
            assert element.count() > 0, f"Field {field} not found"
        
        print("✓ All form fields are present")
        
        # Check for submit button
        submit_btn = page.locator("#predictBtn")
        assert submit_btn.count() > 0, "Submit button not found"
        print("✓ Submit button is present")
        
        browser.close()
        return True
        
    except Exception as e:
        print(f"✗ Form elements test failed: {e}")
        browser.close()
        return False

def test_prediction(playwright):
    """Test the prediction functionality"""
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    
    try:
        page.goto(BASE_URL, wait_until="networkidle")
        
        # Fill form with test data
        page.fill("#indice_org", "75")
        page.select_option("#nivel_educativo", "2")
        page.fill("#pct_mujeres", "40")
        page.fill("#pct_varones", "60")
        page.select_option("#tipo_producto", "Cafe")
        page.fill("#tiempo_ejecucion", "18")
        page.select_option("#brecha_territorial", "media")
        page.fill("#precipitacion", "1200")
        page.fill("#temperatura", "20")
        page.select_option("#sequia", "0")
        
        print("✓ Form filled with test data")
        
        # Submit the form
        page.click("#predictBtn")
        
        # Wait for result
        page.wait_for_selector(".result-card", timeout=10000)
        print("✓ Prediction result displayed")
        
        # Check that prediction contains expected classes
        result_card = page.locator(".result-card")
        result_text = result_card.text_content()
        
        assert any(level in result_text for level in ["Alta", "Media", "Baja"]), \
            f"Unexpected prediction result: {result_text}"
        print(f"✓ Prediction result: {result_text.strip()[:100]}...")
        
        # Check data summary is displayed
        summary = page.locator(".data-summary")
        assert summary.count() > 0, "Data summary not displayed"
        print("✓ Data summary is displayed")
        
        browser.close()
        return True
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        browser.close()
        return False

def test_model_selection(playwright):
    """Test that different models can be selected"""
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    
    try:
        page.goto(BASE_URL, wait_until="networkidle")
        
        models = ["random_forest", "svm", "xgboost"]
        
        for model in models:
            page.select_option("#modelo_ml", model)
            selected = page.locator("#modelo_ml").input_value()
            assert selected == model, f"Model selection failed for {model}"
            print(f"✓ Model {model} can be selected")
        
        browser.close()
        return True
        
    except Exception as e:
        print(f"✗ Model selection test failed: {e}")
        browser.close()
        return False

def run_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("RURAL PRODUCTIVITY CLASSIFIER - TEST SUITE")
    print("=" * 60)
    print()
    
    # Start Flask server
    print("Starting Flask server...")
    server_process = start_flask_server()
    time.sleep(SERVER_STARTUP_TIME)
    print(f"Server started (PID: {server_process.pid})")
    print()
    
    results = []
    
    try:
        with sync_playwright() as playwright:
            # Run tests
            print("-" * 40)
            print("Running tests...")
            print("-" * 40)
            
            results.append(("Page Load", test_page_load(playwright)))
            results.append(("Form Elements", test_form_elements(playwright)))
            results.append(("Prediction", test_prediction(playwright)))
            results.append(("Model Selection", test_model_selection(playwright)))
            
    finally:
        # Stop the server
        print()
        print("Stopping Flask server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped")
    
    # Print summary
    print()
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Total: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
