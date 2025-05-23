import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import App from './App';

// Mock axios
jest.mock('axios');

// Mock environment variables
process.env.REACT_APP_BACKEND_URL = 'http://test-api';

describe('App Component', () => {
  beforeEach(() => {
    // Clear mocks before each test
    jest.clearAllMocks();
  });

  test('renders initial header and input form', () => {
    render(<App />);
    expect(screen.getByText('PickSmart')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ask about products...')).toBeInTheDocument();
  });

  test('handles user input and submission', async () => {
    const mockResponse = {
      data: {
        value: JSON.stringify({
          default: 'Test bot response'
        })
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    render(<App />);
    
    const input = screen.getByPlaceholderText('Ask about products...');
    const submitButton = screen.getByRole('button');

    await userEvent.type(input, 'test query');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('test query')).toBeInTheDocument();
    });
  });

  test('displays error message on API failure', async () => {
    axios.post.mockRejectedValueOnce(new Error('API Error'));

    render(<App />);
    
    const input = screen.getByPlaceholderText('Ask about products...');
    const submitButton = screen.getByRole('button');

    await userEvent.type(input, 'test query');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('An error occurred.')).toBeInTheDocument();
    });
  });

  test('displays product information correctly', async () => {
    const mockResponse = {
      data: {
        value: JSON.stringify({
          products: [{
            title: 'Test Product',
            description: 'Test Description',
            image: 'test.jpg',
            url: 'http://test.com'
          }]
        })
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    render(<App />);
    
    const input = screen.getByPlaceholderText('Ask about products...');
    const submitButton = screen.getByRole('button');

    await userEvent.type(input, 'test query');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Test Product')).toBeInTheDocument();
      expect(screen.getByText('Test Description')).toBeInTheDocument();
      expect(screen.getByRole('link')).toHaveAttribute('href', 'http://test.com');
    });
  });

  test('handles message streaming correctly', async () => {
    const mockResponse = {
      data: {
        value: JSON.stringify({
          default: 'Test streaming message'
        })
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    render(<App />);
    
    const input = screen.getByPlaceholderText('Ask about products...');
    await userEvent.type(input, 'test query');
    fireEvent.submit(screen.getByRole('form'));

    await waitFor(() => {
      expect(screen.getByText(/Test streaming message/)).toBeInTheDocument();
    });
  });
});
